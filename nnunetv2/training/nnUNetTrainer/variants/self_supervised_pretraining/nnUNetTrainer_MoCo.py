import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform, \
    Convert2DTo3DTransform
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.contrastive_view_generator import (
    ContrastiveLearningViewGenerator,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetBaseTrainer,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)

"""
resource: https://github.com/facebookresearch/moco
"""


class nnUNetTrainer_MoCo(nnUNetBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "initial_lr": 0.01,
        "learning_rate_schedule": [120, 160],
        "optimizer_momentum": 0.9,
        "weight_decay": 1e-4,
        "use_cosine_schedule": False,
        "temperature": 0.07,
        "queue_size": 65535,
        "encoder_updating_momentum": 0.999,
        "projection_layer_dimension": 128,
        "num_val_iterations_per_epoch": 0,
    }

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
            **kwargs,
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device, **kwargs
        )

        self.key_projection_layer = None
        self.query_projection_layer = None

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            self.momentum_encoder_network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            self.print_to_log_file("Cloning weights of encoder to momentum encoder")
            for param_q, param_k in tqdm(
                    zip(
                        self.network.parameters(),
                        self.momentum_encoder_network.parameters(),
                    )
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            self.momentum_encoder_network.register_buffer(
                "queue", torch.randn(self.projection_layer_dimension, self.queue_size)
            )

            self.momentum_encoder_network.queue = F.normalize(
                self.momentum_encoder_network.queue, dim=0
            )

            self.momentum_encoder_network.register_buffer(
                "queue_ptr", torch.zeros(1, dtype=torch.long)
            )

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)
                self.momentum_encoder_network = torch.compile(
                    self.momentum_encoder_network
                )

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

                self.momentum_encoder_network = (
                    torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                        self.momentum_encoder_network
                    )
                )
                self.momentum_encoder_network = DDP(
                    self.momentum_encoder_network, device_ids=[self.local_rank]
                )

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def _build_loss(self):
        return torch.nn.CrossEntropyLoss().to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=self.optimizer_momentum
        )

        def lr_lambda(epoch):
            lr = self.initial_lr
            if self.use_cosine_schedule:
                # Cosine learning rate schedule
                lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))
            else:
                # Stepwise learning rate schedule
                for milestone in self.learning_rate_schedule:
                    if epoch >= milestone:
                        lr *= 0.1
                    else:
                        break
            return lr / self.initial_lr

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return optimizer, scheduler

    def forward(self, data, target):

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # due to data augmentation, batch dimension has doubled in size
            # from that, we extract the key and query augmentations back
            key_data, query_data = torch.chunk(data, 2)
            q = self.query_forward(query_data)
            k = self.key_forward(key_data)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum(
                "nc,ck->nk",
                [
                    q,
                    self.momentum_encoder_network.queue.clone()
                    .detach()
                    .to(self.device),
                ],
            )

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            loss = self.loss(logits, labels)
        return logits, loss

    @torch.no_grad()
    def key_forward(self, key_data):
        self._momentum_update_key_encoder()

        if self.is_ddp:
            # TODO: this is not tested
            key_data, idx_unshuffle = self._batch_shuffle_ddp(key_data)
            k = self.momentum_encoder_network(key_data)

            if isinstance(k, list):
                k = k[-1]
            k = k.flatten(start_dim=1)

            if not self.key_projection_layer:
                self.initialize_projection_layers(k.shape[1])
            k = self.key_projection_layer(k)

            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        else:
            k = self.momentum_encoder_network(key_data)

            if isinstance(k, list):
                k = k[-1]
            k = k.flatten(start_dim=1)

            if not self.key_projection_layer:
                self.initialize_projection_layers(k.shape[1])
            k = self.key_projection_layer(k)

            k = F.normalize(k, dim=1)

        return k

    def query_forward(self, query_data):
        q = self.network(query_data)

        if isinstance(q, list):
            q = q[-1]
        q = q.flatten(start_dim=1)

        if not self.query_projection_layer:
            self.initialize_projection_layers(q.shape[1])
        q = self.query_projection_layer(q)

        q = F.normalize(q, dim=1)
        return q

    def initialize_projection_layers(self, in_dimension):
        if not self.query_projection_layer:
            self.query_projection_layer = torch.nn.Linear(
                in_dimension, self.projection_layer_dimension
            ).to(self.device)

        self.key_projection_layer = torch.nn.Linear(
            in_dimension, self.projection_layer_dimension
        ).to(self.device)

        for param_q, param_k in tqdm(
                zip(
                    self.query_projection_layer.parameters(),
                    self.key_projection_layer.parameters(),
                )
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False,  # todo experiment with this
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(
            GaussianBlurTransform(
                (0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )
        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.15
            )
        )
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
                ignore_axes=ignore_axes,
            )
        )
        tr_transforms.append(
            GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)
        )
        tr_transforms.append(
            GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3)
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        tr_transforms.append(NumpyToTensor(["data"], "float"))

        return ContrastiveLearningViewGenerator(base_transforms=Compose(tr_transforms))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.momentum_encoder_network.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.momentum_encoder_network.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.momentum_encoder_network.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.network.parameters(), self.momentum_encoder_network.parameters()
        ):
            param_k.data = (
                    param_k.data * self.encoder_updating_momentum
                    + param_q.data * (1.0 - self.encoder_updating_momentum)
            )

        # we also want to update the weights of the projection layer
        for param_q, param_k in zip(
                self.query_projection_layer.parameters(),
                self.key_projection_layer.parameters(),
        ):
            param_k.data = (
                    param_k.data * self.encoder_updating_momentum
                    + param_q.data * (1.0 - self.encoder_updating_momentum)
            )

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if not self.is_ddp:
            return tensor.clone()

        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
