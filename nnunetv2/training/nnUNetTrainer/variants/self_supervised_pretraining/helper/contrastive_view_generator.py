import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torchio import RandomFlip, RandomAffine, RandomBlur, Compose



class ContrastiveLearningViewGenerator(AbstractTransform):

    def __init__(self, crop_size, base_transforms: Compose = None):
        self.base_transforms = base_transforms if base_transforms else self._get_default_transforms()
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        tensor_data_dict = NumpyToTensor(keys=["data"])(**data_dict)

        cropped_volumes = self._random_crop(tensor_data_dict['data'])
        transformed_volumes = []  # for each batch for each crop apply base_transforms

        for bs in range(cropped_volumes.shape[0]):
            crops = []
            for crop in range(cropped_volumes.shape[1]):
                transformed_volume = self.base_transforms(cropped_volumes[bs, crop])
                crops.append(transformed_volume)

            transformed_volumes.append(torch.stack(crops, dim=0))

        result_dict = {'target': None, 'data': torch.stack(transformed_volumes, dim=0)}
        # batch + augmentations, modality, w, h, d
        # loss will take care of this
        result_dict['data'] = torch.cat(result_dict['data'], dim=0)

        return result_dict

    @staticmethod
    def _get_default_transforms():
        return Compose([
            RandomFlip(),
            RandomAffine(scales=(0.8, 1.2, 0.8, 1.2, 1, 1),
                         degrees=(-10, 10, -10, 10, 0, 0)),
            RandomBlur()
        ])

    """
    Randomly creates two random crops of the input data.
    :param data: torch.Tensor, shape (batch_size, channels, w, h, d)
    :param crop_size: tuple, (w, h, d)
    :return: tensor of shape (batch_size, 2, channels, w, h, d)
    """

    def _random_crop(self, data):
        bs, _, w, h, d = data.shape

        cropped_data = []

        # iterate over batch_dimension
        for i in range(bs):
            crop1_x1 = np.random.randint(0, w - self.crop_size[0])
            crop2_x1 = np.random.randint(0, w - self.crop_size[0])
            crop1_y1 = np.random.randint(0, h - self.crop_size[1])
            crop2_y1 = np.random.randint(0, h - self.crop_size[1])
            crop1_z1 = np.random.randint(0, d - self.crop_size[2])
            crop2_z1 = np.random.randint(0, d - self.crop_size[2])

            crop1_x2 = crop1_x1 + self.crop_size[0]
            crop2_x2 = crop2_x1 + self.crop_size[0]
            crop1_y2 = crop1_y1 + self.crop_size[1]
            crop2_y2 = crop2_y1 + self.crop_size[1]
            crop1_z2 = crop1_z1 + self.crop_size[2]
            crop2_z2 = crop2_z1 + self.crop_size[2]

            crop1 = data[i, :, crop1_x1:crop1_x2, crop1_y1:crop1_y2, crop1_z1:crop1_z2]
            crop2 = data[i, :, crop2_x1:crop2_x2, crop2_y1:crop2_y2, crop2_z1:crop2_z2]

            # TODO: do we need center crop? I don't think so
            # verify that the crops are different
            # view_data = torch.stack((crop1[0], crop2[0]), dim=0)
            # self.view_data(view_data)

            cropped_data.append(torch.stack((crop1, crop2), dim=0))

        return torch.stack(cropped_data, dim=0)

    @staticmethod
    def view_data(data):
        
        from batchviewer import view_batch
        # we can only view 4D data => reduce

        # if data.ndim == 5 (batch, channels, w, h, d) => (batch, w, h, d)
        if data.ndim == 5:
            data = data[:, 0]

        # if data.ndim == 6 (batch, augmentation_views, channels, w, h, d, t) => (augmentation_views, w, h, d)
        if data.ndim == 6:
            data = data[0, :, 0]

        view_batch(data, width=300, height=300)
