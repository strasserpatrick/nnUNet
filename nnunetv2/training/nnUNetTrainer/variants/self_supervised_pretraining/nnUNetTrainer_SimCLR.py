from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast

from torch import nn

from nnunetv2.training.dataloading.contrastive_data_loader import (
    nnUNetContrastiveDataLoader,
)
from nnunetv2.training.dataloading.contrastive_dataset import ContrastiveDataset
from nnunetv2.training.loss.info_nce import InfoNCELoss
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.simclr_transforms import (
    SimCLRTransform,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetBaseTrainer,
)

from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_SimCLR(nnUNetBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "unpack_segmentation": False,
        "temperature": 0.07,
        "learning_rate": 0.05,
        "weight_decay": 1e-4,
        "use_projection_layer": True,
        "latent_space_dim": 512,
        "num_val_iterations_per_epoch": 0,
        "batch_size": 1,
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

        # Encoder feature map to feature vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.class_projection_layer = None

        # simclr specific head
        self.projection_layer = (
            torch.nn.Sequential(
                torch.nn.Linear(self.latent_space_dim, self.latent_space_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.latent_space_dim, 256),
            ).to(self.device)
            if self.use_projection_layer
            else None
        )

        self.info_nce_loss = InfoNCELoss(
            batch_size=self.batch_size,
            device=self.device,
            temperature=self.temperature,
        )

    def forward(self, data, target):

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            data_aug_0 = data
            data_aug_1 = target

            features_0 = self.network(data_aug_0)
            features_1 = self.network(data_aug_1)

            # plain cnn encoder can return list of all feature maps (len=6 for braTS)
            # we are only interesting in the final result
            if isinstance(features_0, list):
                features_0 = features_0[-1]
                features_1 = features_1[-1]

            features_0 = self._extract_feature_vector(features_0)
            features_1 = self._extract_feature_vector(features_1)

            if self.use_projection_layer:
                features_0 = self.projection_layer(features_0)
                features_1 = self.projection_layer(features_1)

            features = torch.cat((features_0, features_1))

            logits, labels = self.info_nce_loss(features=features)
            loss = self.loss(logits, labels)
        return logits, loss

    def _extract_feature_vector(self, feature_map):
        gap_output = self.gap(feature_map)

        if not self.class_projection_layer:
            self.class_projection_layer = nn.Linear(
                gap_output.shape[1], self.latent_space_dim
            ).to(self.device)

        feature_vector = self.class_projection_layer(gap_output)
        return feature_vector

    # OPTIMIZER, SCHEDULER AND LOSS

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.dataloader_train),
            eta_min=0,
            last_epoch=-1,
        )

        return optimizer, scheduler

    def _build_loss(self):
        return torch.nn.CrossEntropyLoss().to(self.device)

    # DATASET, DATA LOADING AND AUGMENTATIONS

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
        return SimCLRTransform()

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        return SimCLRTransform()

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = ContrastiveDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        dataset_val = ContrastiveDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        return dataset_tr, dataset_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        assert dim == 3, "Only 3D is supported"

        # note that the different arguments are not used in the PCRLv2 data loader
        dl_tr = nnUNetContrastiveDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnUNetContrastiveDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )

        return dl_tr, dl_val
