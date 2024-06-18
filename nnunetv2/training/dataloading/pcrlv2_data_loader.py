from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase

import numpy as np


class nnUNetDataLoaderPCRLv2(nnUNetDataLoaderBase):

    def determine_shapes(self):
        return 0, 0

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        global_data = []
        local_data = []

        case_properties = None

        for i in selected_keys:
            glob, loc, properties = self._data.load_case(i)
            global_data.append(glob)
            local_data.append(loc)

            # properties is same for every case in this implementation
            if case_properties is None:
                case_properties = properties

        return {
            "global": np.stack(global_data, axis=0),
            "local": np.stack(local_data, axis=0),
            "properties": case_properties,
            "keys": selected_keys,
        }
