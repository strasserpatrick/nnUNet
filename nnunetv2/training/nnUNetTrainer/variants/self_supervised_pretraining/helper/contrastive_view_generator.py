import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose


class ContrastiveLearningViewGenerator(AbstractTransform):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transforms: Compose, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, **data_dict):
         
        data_augmentations = [self.base_transforms(**data_dict) for _ in range(self.n_views)]
        data_dict['data'] = data_augmentations

        return data_dict