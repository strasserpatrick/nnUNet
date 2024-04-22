from typing import List, Tuple, Union
import numpy as np
import torch
from torch import autocast

import torch.nn.functional as F

from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.contrastive_view_generator import ContrastiveLearningViewGenerator
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from batchgenerators.transforms.abstract_transforms import AbstractTransform


"""
Resources:
https://github.com/sthalles/SimCLR/
https://medium.com/@prabowoyogawicaksana/self-supervised-pre-training-with-simclr-79830997be34
"""

class nnUNetTrainer_SimCLR(nnUNetTrainer):
    DEFAULT_TEMPERATURE_VALUE: float = 0.07
    INITIAL_LEARNING_RATE: float = 0.0003
    WEIGHT_DECAY: float = 1e-4

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
            **kwargs
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        if not self.fold == "all":
            print("Warning: Using SSL pretraining with a single fold. This is not recommended.")

        self.temperature = self.DEFAULT_TEMPERATURE_VALUE if "temperature" not in kwargs else kwargs['temperature']
        self.initial_learning_rate = self.INITIAL_LEARNING_RATE if "initial_learning_rate" not in kwargs else kwargs['initial_learning_rate']
        self.weight_decay = self.WEIGHT_DECAY if "weight_decay" not in kwargs else kwargs["weight_decay"]
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(),
                          lr=self.initial_learning_rate,
                          weight_decay=self.weight_decay,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.dataloader_train), eta_min=0,
                                                           last_epoch=-1)
        return optimizer, scheduler

    def _build_loss(self):
        return torch.nn.CrossEntropyLoss().to(self.device)

        
    def forward(self, data, target):

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            # TODO: how do I get this right that I do not confuse bs with augmentations in one dimension?

            features = self.network.encoder(data)

            logits, labels = self.__info_nce_loss(features)
            loss = self.loss(logits, labels)
        return loss
    

    def __info_nce_loss(self, features):

        n_views = 2

        labels = torch.cat([torch.arange(self.batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
    # disabling nnUNet data augmentation and replacing by SimCLR 
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inferene!
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 1,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        
        # TODO: concrete simCLR augmentation parameters here
        ssl_transforms = [(GaussianNoiseTransform())]
        ssl_transforms.append(NumpyToTensor(['data'], 'float'))



        return ContrastiveLearningViewGenerator(base_transforms=Compose(ssl_transforms))
    


        