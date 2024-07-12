import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torchio.transforms import (
    Compose,
    RandomFlip,
    RandomAffine,
    RandomBlur,
)


class MedicalSimCLRTransform(AbstractTransform):
    def __init__(self):
        self.contrastive_transforms = Compose([
            RandomFlip(),
            RandomAffine(scales=(0.8, 1.2, 0.8, 1.2, 1, 1),
                         degrees=(-10, 10, -10, 10, 0, 0)),
            RandomBlur()
        ])

    def __call__(self, **data_dict):
        converted_data_dict = NumpyToTensor(keys=["global", "local"])(**data_dict)
        return self.transform(**converted_data_dict)

    def transform(self, **data_dict):
        data = data_dict["global"]

        img_1 = data[:, 0]
        img_2 = data[:, 1]

        # iterate over batch dimension because torchio only allows 4D tensors
        # https://github.com/fepegar/torchio/discussions/562
        input_1 = torch.stack([self.contrastive_transforms(inst) for inst in img_1])
        input_2 = torch.stack([self.contrastive_transforms(inst) for inst in img_2])

        return {
            "input_1": input_1,
            "input_2": input_2,
        }
