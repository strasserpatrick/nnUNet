import os
from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class PCRLv2Dataset(nnUNetDataset):

    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):

        super().__init__(folder, case_identifiers, num_images_properties_loading_threshold,
                         folder_with_segs_from_previous_stage)
        # print('loading dataset')
        if case_identifiers is None:
            file_names = [i for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
            case_identifiers = ["_".join(i.split("_")[:-2]) for i in file_names]
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['global_view_files'] = [join(folder, f"{c}_global_{i}.npz") for i in range(6)]
            self.dataset[c]['local_view_files'] = [join(folder, f"{c}_local_{i}.npz") for i in range(6)]
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')


    def load_case(self, key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        else:
            # data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            data = {"global": [], "local": []}

            for file in entry['global_view_files']:
                data["global"].append(np.load(file)['data'])
            for file in entry['local_view_files']:
                data["local"].append(np.load(file)['data'])

            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')

        return data, None, entry['properties']
