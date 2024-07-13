from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from torch import autocast

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.contrastive_view_generator import \
    ContrastiveLearningViewGenerator
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.loss_functions import (
    info_nce_loss,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetSSLBaseTrainer,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from torch.nn.parallel import DistributedDataParallel as DDP


"""
Resources:
https://github.com/sthalles/SimCLR/
https://medium.com/@prabowoyogawicaksana/self-supervised-pre-training-with-simclr-79830997be34
"""


class nnUNetTrainer_SimCLR(nnUNetSSLBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "temperature": 0.07,
        "initial_learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "use_projection_layer": True,
        "latent_space_dim": 8096,
        "num_val_iterations_per_epoch": 0,
        "batch_size": 8,
        "num_epochs": 100,
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

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            num_dims = self._determine_out_dimensionality()
            self._initialize_projection_layer(num_dims)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
                self.projection_layer = DDP(self.projection_layer, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

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
            # we have input of form (batch_size, augmentation_views, c, x, y, z)
            # first we flatten the augmentation views into the batch dimension
            combined_data = data.view(-1, *data.shape[2:])
            features = self.network(combined_data)

            # plain cnn encoder returns list of all feature maps (len=6 for braTS)
            # we are only interested in the final result
            if isinstance(features, list):
                features = features[-1]

            flattened_features = torch.flatten(self.gap(features), 1, -1)

            # this only has effect if you use_projection_layer is True
            projected_features = self.projection_layer(flattened_features)

            logits, labels = info_nce_loss(
                features=projected_features,
                batch_size=self.batch_size,
                device=self.device,
                temperature=self.temperature,
            )
            loss = self.loss(logits, labels)
        return logits, loss

    def _project_features(self, features_0, features_1):
        if self.use_projection_layer:
            features_0 = self.projection_layer(features_0)
            features_1 = self.projection_layer(features_1)

        return features_0, features_1

    def _initialize_projection_layer(self, in_dimension):
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(in_dimension, self.latent_space_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_space_dim, self.latent_space_dim),
            torch.nn.ReLU(),
        ).to(self.device)

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
        return ContrastiveLearningViewGenerator(crop_size=patch_size)
