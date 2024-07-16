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
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


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

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            full_model = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.num_input_channels, # output channels = input channels for MG as we mimic an autoencoder
                self.enable_deep_supervision
            ).to(self.device)

            self.network = full_model.encoder
            self.decoder = full_model.decoder

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
                self.decoder = torch.compile(self.decoder)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

                self.decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
                self.decoder = DDP(self.decoder, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _build_loss(self):
        return torch.nn.MSELoss()

    def configure_optimizers(self):
        all_params = list(self.network.parameters()) + list(self.decoder.parameters())
        return torch.optim.SGD(all_params, self.learning_rate, momentum=self.sgd_momentum), None

    def forward(self, data, target):
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            f = self.network(data) # this is the encoder
            logits = self.decoder(f) # and decoding it back to full size

            if isinstance(logits, list):
                logits = logits[-1]
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
