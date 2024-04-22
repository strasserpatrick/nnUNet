import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose


class ContrastiveLearningViewGenerator(AbstractTransform):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transforms: Compose, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, **data_dict):
         
        result_dict = {'target': None, 'data': []}
        for _ in range(self.n_views):
            aug_view_dict = self.base_transforms(**data_dict)
            result_dict['data'].append(aug_view_dict['data'])
        
        result_dict['data'] = torch.stack(result_dict['data'], dim=0) # batch, augmentations, modality, w, h, d

        return result_dict