from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

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
        "initial_lr": 0.03,  # TODO: momentum and learning rate scheduler in moco implementation?
        "temperature": 0.07,
        "num_val_iterations_per_epoch": 0,
        "queue_size": 65535,
        "encoder_updating_momentum": 0.999,
        "use_projection_layer": True,
        "projection_layer_dimension": 128,
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

        self.query_projection_layer = None
        self.key_projection_layer = None

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

            self.print_to_log_file("Cloning weights of encoder to momentum decoder")
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
        )

        # this is a scheduler that does not schedule
        # TODO: improve moco code
        lambda_func = lambda epoch: 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)

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
        return loss

    @torch.no_grad
    def key_forward(self, key_data):
        self._momentum_update_key_encoder()

        if self.is_ddp:
            # TODO: this is not tested
            key_data, idx_unshuffle = self._batch_shuffle_ddp(key_data)
            k = self.momentum_encoder_network(key_data)

            if isinstance(k, list):
                k = k[-1]
            k = k.flatten(start_dim=1)

            if self.use_projection_layer:
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

            if self.use_projection_layer:
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

        if self.use_projection_layer:
            if not self.query_projection_layer:
                self.initialize_projection_layers(q.shape[1])
            q = self.query_projection_layer(q)

        q = F.normalize(q, dim=1)
        return q

    def initialize_projection_layers(self, in_dimension):
        if self.use_projection_layer:
            if not self.query_projection_layer:
                self.query_projection_layer = torch.nn.Linear(
                    in_dimension, self.projection_layer_dimension
                ).to(self.device)

                self.key_projection_layer = torch.nn.Linear(
                    in_dimension, self.projection_layer_dimension
                ).to(self.device)

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 1,
        order_resampling_seg: int = 0,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:

        # TODO: concrete MoCo augmentation parameters here
        ssl_transforms = [(GaussianNoiseTransform())]
        ssl_transforms.append(NumpyToTensor(["data"], "float"))

        return ContrastiveLearningViewGenerator(base_transforms=Compose(ssl_transforms))

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.momentum_encoder_network.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

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
        if self.use_projection_layer:
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
