from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
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
        converted_data_dict = NumpyToTensor(keys=["global", "local"])(**data_dict)
        return self.pclr_transform(**converted_data_dict)

    def pclr_transform(self, **data_dict):
        global_data = data_dict["global"]
        local_data = data_dict["local"]

        global_img_1 = global_data[:, 0]
        global_img_2 = global_data[:, 1]

        # iterate over batch dimension because torchio only allows 4D tensors
        # https://github.com/fepegar/torchio/discussions/562
        input_1 = torch.stack([self.spatial_transforms(inst) for inst in global_img_1])
        input_2 = torch.stack([self.spatial_transforms(inst) for inst in global_img_2])

        gt_1 = copy.deepcopy(input_1)
        gt_2 = copy.deepcopy(input_2)

        input_1 = torch.stack([self.global_transforms(inst) for inst in input_1])
        input_2 = torch.stack([self.global_transforms(inst) for inst in input_2])

        local_inputs = []
        for i in range(local_data.shape[1]):
            local_img = local_data[:, i]
            local_img = torch.stack([self.spatial_transforms(inst) for inst in local_img])
            local_img = torch.stack([self.local_transforms(inst) for inst in local_img])
            local_inputs.append(local_img)

        return {
            "input_1": input_1,
            "input_2": input_2,
            "gt_1": gt_1,
            "gt_2": gt_2,
            "local_inputs": torch.stack(local_inputs),
            "properties": data_dict["properties"],
            "keys": data_dict["keys"],
        }
