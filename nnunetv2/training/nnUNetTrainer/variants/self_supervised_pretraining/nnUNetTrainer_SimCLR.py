from typing import List, Tuple, Union

import numpy as np

from nnunetv2.training.dataloading.contrastive_data_loader import nnUNetContrastiveDataLoader
from nnunetv2.training.dataloading.contrastive_dataset import ContrastiveDataset
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.simclr_transforms import \
    SimCLRTransform
from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.ssl_base_trainer import \
    nnUNetBaseTrainer

from batchgenerators.transforms.abstract_transforms import AbstractTransform


class nnUNetTrainer_SimCLR(nnUNetBaseTrainer):

    def forward(self, data, target):
        print("hello")

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

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = ContrastiveDataset(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)
        dataset_val = ContrastiveDataset(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        assert dim == 3, "Only 3D is supported"

        # note that the different arguments are not used in the PCRLv2 data loader
        dl_tr = nnUNetContrastiveDataLoader(dataset_tr, self.batch_size,
                                            initial_patch_size,
                                            self.configuration_manager.patch_size,
                                            self.label_manager,
                                            oversample_foreground_percent=self.oversample_foreground_percent,
                                            sampling_probabilities=None, pad_sides=None)
        dl_val = nnUNetContrastiveDataLoader(dataset_val, self.batch_size,
                                             self.configuration_manager.patch_size,
                                             self.configuration_manager.patch_size,
                                             self.label_manager,
                                             oversample_foreground_percent=self.oversample_foreground_percent,
                                             sampling_probabilities=None, pad_sides=None)

        return dl_tr, dl_val
