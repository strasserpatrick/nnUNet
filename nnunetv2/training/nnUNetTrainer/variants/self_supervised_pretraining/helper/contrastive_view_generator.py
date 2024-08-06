import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torchio import RandomFlip, RandomAffine, RandomBlur, Compose

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.transform_utils import crop_data


class ContrastiveLearningViewGenerator(AbstractTransform):

    def __init__(self, crop_size, base_transforms: Compose = None, n_views: int = 2):
        self.base_transforms = base_transforms if base_transforms else self._get_default_transforms()
        self.crop_size = crop_size
        self.n_views = n_views

    def __call__(self, **data_dict):
        tensor_data_dict = NumpyToTensor(keys=["image"])(**data_dict)

        views_list = [crop_data(tensor_data_dict["image"], self.crop_size, center_crop=True) for _ in
                      range(self.n_views)]
        cropped_volumes = torch.stack(views_list, dim=0)  # batch, views, channels, w, h, d

        result_crops = []
        for crop_idx in range(cropped_volumes.shape[0]):
            transformed_volume = self.base_transforms(cropped_volumes[crop_idx])
            result_crops.append(transformed_volume)

        result_dict = {'segmentation': None, 'image': torch.stack(result_crops, dim=0)}
        # batch, augmentation views, channels , w, h, d

        return result_dict

    @staticmethod
    def _get_default_transforms():
        return Compose([
            RandomFlip(),
            RandomAffine(scales=(0.8, 1.2, 0.8, 1.2, 1, 1),
                         degrees=(-10, 10, -10, 10, 0, 0)),
            RandomBlur()
        ])
