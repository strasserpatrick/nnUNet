from typing import List, Tuple, Union
import numpy as np
import torch
from torch import autocast

import torch.nn.functional as F
import torch.nn

from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from tqdm import tqdm

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.contrastive_view_generator import (
    ContrastiveLearningViewGenerator,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from torch.nn.parallel import DistributedDataParallel as DDP

"""
resource: https://github.com/facebookresearch/moco
"""


class nnUNetTrainer_MoCo(nnUNetTrainer):

    DEFAULT_PARAMS: dict = {
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
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )

        if not self.fold == "all":
            print(
                "Warning: Using SSL pretraining with a single fold. This is not recommended."
            )

        self._set_hyperparameters(**kwargs)

    def _set_hyperparameters(self, **kwargs):
        for attribute_name in self.DEFAULT_PARAMS:
            if attribute_name in kwargs:
                # overwrite
                setattr(self, attribute_name, kwargs[attribute_name])
            else:
                # default value
                setattr(self, attribute_name, self.DEFAULT_PARAMS[attribute_name])

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

            self.momentum_encoder_network.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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

    def forward(self, data, target):

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # due to data augmentation, batch dimension has doubled in size
            # to keep required gpu memory the same, forward pass is done in two passes.
            data_aug_0, data_aug_1 = torch.chunk(data, 2)

            features_0 = self.network.forward(data_aug_0)
            features_1 = self.network.forward(data_aug_1)

            # plain cnn encoder returns list of all feature maps (len=6 for braTS)
            # we are only interesting in the final result
            if isinstance(features_0, list):
                features_0 = features_0[-1]
                features_1 = features_1[-1]

            features = torch.cat((features_0, features_1)).flatten(start_dim=1)

            if self.use_projection_layer:

                # dynamic initialization depending on encoders output shape
                if not self.projection_layer:
                    self.projection_layer = torch.nn.Linear(
                        features.shape[1], self.latent_space_dim
                    ).to(self.device)

                features = self.projection_layer(features)

            logits, labels = self.__info_nce_loss(features)
            loss = self.loss(logits, labels)
        return loss

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
