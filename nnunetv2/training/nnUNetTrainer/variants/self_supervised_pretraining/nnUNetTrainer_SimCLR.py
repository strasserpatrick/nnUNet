from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform,
    NumpyToTensor,
)
from torch import autocast

from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetBaseTrainer,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.contrastive_view_generator import (
    ContrastiveLearningViewGenerator,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.loss_functions import (
    info_nce_loss,
)
from nnunetv2.utilities.helpers import dummy_context

"""
Resources:
https://github.com/sthalles/SimCLR/
https://medium.com/@prabowoyogawicaksana/self-supervised-pre-training-with-simclr-79830997be34
"""


class nnUNetTrainer_SimCLR(nnUNetBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "temperature": 0.07,
        "initial_learning_rate": 0.0001,
        "weight_decay": 1e-4,
        "use_projection_layer": True,
        "latent_space_dim": 8096,
        "num_val_iterations_per_epoch": 0,
        "batch_size": 8,
        "num_epochs": 5000,
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

        self.projection_layer = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.initial_learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(self.dataloader_train), eta_min=0, last_epoch=-1
        )
        return optimizer, scheduler

    def _build_loss(self):
        return torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, data, target):

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # due to data augmentation, batch dimension has doubled in size
            # to keep required gpu memory the same, forward pass is done in two passes.
            data_aug_0, data_aug_1 = torch.chunk(data, 2)

            features_0 = self.network(data_aug_0)

            with torch.no_grad():
                features_1 = self.network(data_aug_1)

            # plain cnn encoder returns list of all feature maps (len=6 for braTS)
            # we are only interesting in the final result
            if isinstance(features_0, list):
                features_0 = features_0[-1]
                features_1 = features_1[-1]

            features_0 = features_0.flatten(start_dim=1)
            features_1 = features_1.flatten(start_dim=1)

            if self.use_projection_layer:

                # dynamic initialization depending on encoders output shape
                if not self.projection_layer:
                    self.projection_layer = torch.nn.Sequential(
                        torch.nn.Linear(features_1.shape[1], self.latent_space_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.latent_space_dim, self.latent_space_dim),
                        torch.nn.ReLU(),
                    ).to(self.device)

                features_0 = self.projection_layer(features_0)
                features_1 = self.projection_layer(features_1)

            features = torch.cat((features_0, features_1))

            logits, labels = info_nce_loss(
                features=features,
                batch_size=self.batch_size,
                device=self.device,
                temperature=self.temperature,
            )
            loss = self.loss(logits, labels)
        return logits, loss

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
