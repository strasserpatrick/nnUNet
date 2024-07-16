from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from torch import autocast

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.mg_transforms import MGTransform
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetSSLBaseTrainer,
)
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_MG(nnUNetSSLBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "num_val_iterations_per_epoch": 0,
        "learning_rate": 1e-3,
        "sgd_momentum": 0.9,
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

    def _build_loss(self):
        return torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.SGD(self.network.parameters(), self.learning_rate, momentum=self.sgd_momentum), None

    def forward(self, data, target):
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            logits = self.network(data)
            loss = self.loss(logits, target)

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
        return MGTransform(crop_size=patch_size)
