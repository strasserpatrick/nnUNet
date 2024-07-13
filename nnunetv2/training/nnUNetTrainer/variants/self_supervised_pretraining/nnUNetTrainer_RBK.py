from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.rbk_transforms import (
    RBKTransform,
)
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import (
    nnUNetSSLBaseTrainer,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)


class nnUNetTrainer_RBK(nnUNetSSLBaseTrainer):
    DEFAULT_PARAMS: dict = {
        "num_val_iterations_per_epoch": 0,
        "num_epochs": 1000,
        "batch_size": 8,
        "feature_dimension": 64,
        "order_n_class": 10,  # we need as many classes as total permutations to guess from
        "num_cubes_per_side": 2,
        "learning_rate": 1e-3,
        "learning_rate_decay": [250],
        "weight_decay": 1e-6,
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

        self.num_cubes = self.num_cubes_per_side**3

        # dynamic feature extractor initialization
        self.gap = nn.AdaptiveAvgPool3d(1).to(self.device)
        self.feature_extractor = None

        self.order_fc = nn.Sequential(
            nn.Linear(self.num_cubes * self.feature_dimension, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.order_n_class),
        ).to(self.device)

        # module for predicting horizontal rotation (classification task)
        self.ver_rot_fc = nn.Sequential(
            nn.Linear(self.num_cubes * self.feature_dimension, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_cubes),
        ).to(self.device)

        # module for predicting vertical rotation (classification task)
        self.hor_rot_fc = nn.Sequential(
            nn.Linear(self.num_cubes * self.feature_dimension, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_cubes),
        ).to(self.device)

        self.sigmoid = torch.nn.Sigmoid().to(self.device)

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
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            out_dim = self._determine_out_dimensionality()
            self._initialize_feature_extractor(out_dim)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

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
            # data has shape [BatchSize, Cubes, Channels, Height, Width, Depth]
            # we transform this to loop over cubes
            data = data.transpose(0, 1)
            feature_vectors = self._extract_feature_vectors(data)

            order_logits = self.order_fc(feature_vectors)
            hor_rot_logits = self.hor_rot_fc(feature_vectors)
            ver_rot_logits = self.ver_rot_fc(feature_vectors)

        loss = self._compute_rbk_loss(
            order_logits, hor_rot_logits, ver_rot_logits, target
        )
        return feature_vectors, loss

    def _compute_rbk_loss(self, order_logits, hor_rot_logits, ver_rot_logits, target):
        order_loss_fn, rotate_loss_fn = self.loss

        order_loss = order_loss_fn(order_logits, target["order_label"])

        hor_rot_loss = rotate_loss_fn(
            hor_rot_logits, target["hor_label"].to(torch.float16)
        )
        ver_rot_loss = rotate_loss_fn(
            ver_rot_logits, target["ver_label"].to(torch.float16)
        )
        rot_loss = hor_rot_loss + ver_rot_loss

        return (order_loss + rot_loss) / 2

    def _extract_feature_vectors(self, cubes):
        feature_vectors = []
        for cube in cubes:
            conv_x = self.network(cube)

            dense_x = self.gap(conv_x)
            dense_x = torch.flatten(dense_x, 1, -1)

            logits = self.feature_extractor(dense_x)
            feature_vectors.append(logits)

        feature_vectors = torch.cat(feature_vectors, 1)
        return feature_vectors

    def _determine_out_dimensionality(self):
        # compute from patch size by running a nograd run
        x, y, z = self.configuration_manager.patch_size
        random_ipt = torch.randn((1, self.num_input_channels, x, y, z)).to(self.device)

        with torch.no_grad():
            output = self.gap(self.network(random_ipt))

        flattened_output = torch.flatten(output, 1, -1)
        return flattened_output.shape[1]

    def _initialize_feature_extractor(self, input_nodes):
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_nodes, self.feature_dimension),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.feature_dimension),
        ).to(self.device)

    # OPTIMIZER, SCHEDULER AND LOSS

    def configure_optimizers(self):
        all_parameters = (
            list(self.network.parameters())
            + list(self.feature_extractor.parameters())
            + list(self.order_fc.parameters())
            + list(self.ver_rot_fc.parameters())
            + list(self.hor_rot_fc.parameters())
        )

        optimizer = torch.optim.Adam(
            all_parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.learning_rate_decay
        )

        return optimizer, scheduler

    def _build_loss(self):
        order_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean").to(self.device)
        rotate_loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)

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
