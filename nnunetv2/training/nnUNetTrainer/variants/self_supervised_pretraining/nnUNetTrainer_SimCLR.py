from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from torch import autocast

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
        "initial_learning_rate": 0.0003,
        "weight_decay": 1e-4,
        "use_projection_layer": True,
        "latent_space_dim": 8096,
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

        self.temperature = None
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
        order_resampling_data: int = 1,
        order_resampling_seg: int = 0,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:

        ssl_transforms = [
            RandomCropTransform(),
            GaussianNoiseTransform(p_per_sample=0.8, p_per_channel=0.8),
            GaussianBlurTransform(p_per_sample=0.8, p_per_channel=0.8),
            MirrorTransform(p_per_sample=0.3),
            NumpyToTensor(["data"], "float"),
        ]

        return ContrastiveLearningViewGenerator(base_transforms=Compose(ssl_transforms))
