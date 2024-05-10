from time import time
from typing import List

import numpy as np
import torch
import torch.nn
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetBaseTrainer(nnUNetTrainer):

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
        if not hasattr(self, "DEFAULT_PARAMS"):
            print("Warning: DEFAULT_PARAMS not set in nnUNetBaseTrainer")
            return
        for attribute_name in self.DEFAULT_PARAMS:
            if attribute_name in kwargs:
                # overwrite
                setattr(self, attribute_name, kwargs[attribute_name])
            else:
                # default value
                setattr(self, attribute_name, self.DEFAULT_PARAMS[attribute_name])

    # disabling nnUNet data augmentation and replacing by SimCLR
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inferene!
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def set_deep_supervision_enabled(self, enabled: bool):
        # we have deep supervision disabled, but as we do not have a decoder here,
        # we have to overwrite this method
        pass

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        pass

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
