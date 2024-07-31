from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torchio import RandomFlip, RandomAffine, RandomBlur, Compose, RandomNoise

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.contrastive_view_generator import (
    ContrastiveLearningViewGenerator,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetSSLBaseTrainer,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)


class nnUNetTrainer_BYOL(nnUNetSSLBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "num_epochs": 2000,
        "learning_rate": 1e-4,
        # "sgd_momentum": 0.9,
        "weight_decay": 1e-4,
        "hidden_dim": 256,
        "pred_dim": 256,
        "momentum": 0.996,
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

        # key encoder is target network -> projection layer
        self.key_projection_layer = None

        # query encoder is online network -> projection layer -> prediction layer
        self.query_projection_layer = None
        # no_grad is set when we momentum update the key encoder
        self.query_prediction_layer = torch.nn.Sequential(
            torch.nn.Linear(self.pred_dim, self.hidden_dim),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_dim, self.pred_dim)).to(self.device)

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

            self.target_network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            out_dim = self._determine_out_dimensionality()
            self._initialize_projection_layers(out_dim)

            self.print_to_log_file("Cloning weights of query to key encoder and disabling gradient updates")
            for param_q, param_k in tqdm(
                    zip(
                        self.network.parameters(),
                        self.target_network.parameters(),
                    )
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)
                self.target_network = torch.compile(self.target_network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.target_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.target_network
                )
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

                self.query_projection_layer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.query_projection_layer
                )

                self.query_projection_layer = DDP(self.query_projection_layer, device_ids=[self.local_rank])

                self.query_prediction_layer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.query_prediction_layer
                )
                self.query_prediction_layer = DDP(self.query_prediction_layer, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def _build_loss(self):
        return torch.nn.CosineSimilarity(dim=1)

    def configure_optimizers(self):
        all_params = list(self.network.parameters()) + list(self.query_projection_layer.parameters()) + list(
            self.key_projection_layer.parameters()) + list(self.query_prediction_layer.parameters())

        optim = torch.optim.Adam(all_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        return optim, None

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # update key encoder
        for param_q, param_k in zip(self.network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        # update key projection layer
        for param_q, param_k in zip(self.query_projection_layer.parameters(), self.key_projection_layer.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            param_k.requires_grad = False

    def forward(self, data, target):

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            view1, view2 = data[:, 0], data[:, 1]

            self._momentum_update_key_encoder()

            # first view
            f1 = torch.flatten(self.gap(self.network(view1)), 1, -1)
            p1 = self.query_prediction_layer(self.query_projection_layer(f1))

            with torch.no_grad():
                f2 = torch.flatten(self.gap(self.target_network(view2)), 1, -1)
                z2 = self.key_projection_layer(f2)

            # reverse views
            f2 = torch.flatten(self.gap(self.network(view2)), 1, -1)
            p2 = self.query_prediction_layer(self.query_projection_layer(f2))

            with torch.no_grad():
                f1 = torch.flatten(self.gap(self.target_network(view1)), 1, -1)
                z1 = self.key_projection_layer(f1)

            loss = -2 * (self.loss(p1, z2).mean() + self.loss(p2, z1).mean())
        return f2, loss

    def _initialize_projection_layers(self, in_dimension):
        self.query_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(in_dimension, self.hidden_dim),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_dim, self.pred_dim),
        ).to(self.device)

        self.key_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(in_dimension, self.hidden_dim),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_dim, self.pred_dim),
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
        transforms = Compose([
            RandomFlip(),
            RandomAffine(scales=(0.8, 1.2, 0.8, 1.2, 1, 1),
                         degrees=(-10, 10, -10, 10, 0, 0)),

            RandomBlur(),
            RandomNoise()
        ])
        return ContrastiveLearningViewGenerator(crop_size=patch_size, base_transforms=transforms)
