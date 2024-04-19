import torch
from torch import autocast

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_SSLPretraining(nnUNetTrainer):
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

        
    def forward(self, data, target):
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data) # we can access the self.network.encoder here instead, super nice
            # del data
            loss = self.loss(output, target)
        return loss