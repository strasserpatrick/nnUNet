import torch

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

        if not "ssl_strategy" in kwargs:
            raise ValueError("Please specify the SSL strategy when using the SSL pretraining trainer.")

        if not self.fold == "all":
            print("Warning: Using SSL pretraining with a single fold. This is not recommended.")

        self.ssl_strategy = kwargs["ssl_strategy"]


        # ssl strategy influences the loss function, optimizer and scheduler, get_training_transformes and train_step