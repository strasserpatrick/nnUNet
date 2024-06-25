import math
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.dataloading.contrastive_data_loader import nnUNetContrastiveDataLoader
from nnunetv2.training.dataloading.contrastive_dataset import ContrastiveDataset
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.loss_functions import info_nce_loss
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.rbk_transforms import RBKTransform
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.simclr_transforms import \
    SimCLRTransform
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import \
    nnUNetSSLBaseTrainer

from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nnunetv2.utilities.helpers import dummy_context


class nnUNetSSLTrainer_RBK(nnUNetSSLBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "num_val_iterations_per_epoch": 0,
        "num_epochs": 100,
        "feature_dimension": 64,
        "order_n_class": 100,  # permutations have hemming distance of 100 -> 100 classes
        "num_cubes_per_side": 2,
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

        self.num_cubes = self.num_cubes_per_side ** 3

        # dynamic feature extractor initialization
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.feature_extractor = None

        self.order_fc = nn.Sequential(
            nn.Linear(self.num_cubes * 64, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, self.order_n_class)
        )

        # module for predicting horizontal rotation (classification task)
        self.ver_rot_fc = nn.Sequential(
            nn.Linear(self.num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_cubes))

        # module for predicting vertical rotation (classification task)
        self.hor_rot_fc = nn.Sequential(
            nn.Linear(self.num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_cubes))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data, target):

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # data has shape [BatchSize, Cubes, Channels, Height, Width, Depth]
            # we transform this to loop over cubes
            data = data.transpose(0, 1)
            feature_vectors = self._extract_feature_vectors(data)

            order_logits = self.order_fc(feature_vectors)
            hor_rot_logits = self.sigmoid(self.hor_rot_fc(feature_vectors))
            ver_rot_logits = self.sigmoid(self.ver_rot_fc(feature_vectors))

            loss = self._compute_rbk_loss(order_logits, hor_rot_logits, ver_rot_logits, target)
            return loss

    def _compute_rbk_loss(self, order_logits, hor_rot_logits, ver_rot_logits, target):
        order_gt, hor_rot_gt, ver_rot_gt = target
        order_loss_fn, rotate_loss_fn = self.loss

        order_loss = order_loss_fn(order_logits, order_gt)
        rot_loss = rotate_loss_fn(hor_rot_logits, hor_rot_gt) + rotate_loss_fn(
            ver_rot_logits, ver_rot_gt)

        return (order_loss + rot_loss) / 2

    def _extract_feature_vectors(self, data):
        feature_vectors = []
        for cube in data:
            conv_x = self.network(cube)
            dense_x = self.gap(conv_x)

            # TODO: is this necessary?
            dense_x = torch.flatten(dense_x, 1, -1)

            if not self.feature_extractor:
                self._initialize_feature_extractor(dense_x.shape[1])

            logits = self.feature_extractor(dense_x)
            feature_vectors.append(logits)

        feature_vectors = torch.cat(feature_vectors, 1)
        return feature_vectors

    def _initialize_feature_extractor(self, input_nodes):
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_nodes, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
        )

    # OPTIMIZER, SCHEDULER AND LOSS

    def configure_optimizers(self):
        ...

    def _build_loss(self):
        order_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
        rotate_loss_fn = nn.BCELoss(reduction='mean').to(self.device)

        return order_loss_fn, rotate_loss_fn

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
        return RBKTransform(crop_size=patch_size)
