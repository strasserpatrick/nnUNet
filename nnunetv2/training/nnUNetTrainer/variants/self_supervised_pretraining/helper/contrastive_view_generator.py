import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose


class ContrastiveLearningViewGenerator(AbstractTransform):

    def __init__(self, base_transforms: Compose, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, **data_dict):
         
        result_dict = {'target': None, 'data': []}
        for _ in range(self.n_views):
            aug_view_dict = self.base_transforms(**data_dict)
            result_dict['data'].append(aug_view_dict['data'])
        
        assert not torch.equal(*result_dict['data'])
        result_dict['data'] = torch.cat(result_dict['data'], dim=0) 
        # batch + augmentations, modality, w, h, d
        # loss will take care of this

        return result_dict