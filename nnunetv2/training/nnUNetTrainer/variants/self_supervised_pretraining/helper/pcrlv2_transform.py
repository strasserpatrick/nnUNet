from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from torchio.transforms import (
    Compose,
    RandomFlip,
    RandomAffine,
    RandomBlur,
    RandomNoise,
    RandomGamma,
    ZNormalization,
    RandomSwap,
)

import copy
import numpy as np
import torch

class PCRLv2Transform(AbstractTransform):
    def __init__(self):
        self.spatial_transforms = Compose(
            [
                RandomFlip(),
                RandomAffine(),
            ]
        )

        self.local_transforms = Compose(
            [
                RandomBlur(),
                RandomNoise(),
                RandomGamma(),
                ZNormalization(),
            ]
        )

        self.global_transforms = Compose(
            [
                RandomBlur(),
                RandomNoise(),
                RandomGamma(),
                RandomSwap(patch_size=(8, 4, 4)),
                ZNormalization(),
            ]
        )

    def __call__(self, **data_dict):

        # extract data
        global_data = data_dict['data']['global']
        local_data = data_dict['data']['local']

        global_img_1 = np.expand_dims(global_data[0], axis=0)
        global_img_2 = np.expand_dims(global_data[1], axis=0)

        input_1 = self.spatial_transforms(global_img_1)
        input_2 = self.spatial_transforms(global_img_2)

        gt_1 = copy.deepcopy(input_1)
        gt_2 = copy.deepcopy(input_2)

        input1 = self.global_transforms(input1)
        input2 = self.global_transforms(input2)

        local_input = []
        for i in range(local_data.shape[0]):
            local_img = np.expand_dims(local_data[i], axis=0)
            local_img = self.spatial_transforms(local_img)
            local_img = self.local_transforms(local_img)
            local_input.append(local_img)

        return (
            torch.tensor(input_1, dtype=torch.float),
            torch.tensor(input_2, dtype=torch.float),
            torch.tensor(gt_1, dtype=torch.float),
            torch.tensor(gt_2, dtype=torch.float),
            local_input,
        )
