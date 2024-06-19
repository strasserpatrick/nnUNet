import os
from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class ContrastiveDataset(nnUNetDataset):

    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):

        super().__init__(folder, case_identifiers, num_images_properties_loading_threshold,
                         folder_with_segs_from_previous_stage)
        case_identifiers = self._load_case_identifiers(folder)

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['global_view_file'] = join(folder, f"{c}.npz")
            local_cid = c.replace("global", "local")
            self.dataset[c]['local_view_file'] = join(folder, f"{local_cid}.npz")
            properties_cid = "_".join(c.split("_")[:-2])
            self.dataset[c]['properties_file'] = join(folder, f"{properties_cid}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

    def _load_case_identifiers(self, folder):
        case_identifiers = []
        for fn in os.listdir(folder):
            if fn.endswith(".npz") and (fn.find("segFromPrevStage") == -1) and (fn.find("global") != -1):
                cid = fn[:-4]
                case_identifiers.append(cid)
        return case_identifiers 

    def load_case(self, key):
        # we load global as data and local as seg to keep the flow
        entry = self[key]
        if 'open_data_file' in entry.keys():
            global_data = entry['open_data_file']
        else:
            global_data = np.load(entry["global_view_file"], 'r')['data']
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = global_data

        if 'open_seg_file' in entry.keys():
            local_data = entry['open_seg_file']
        else:
            local_data = np.load(entry["local_view_file"], 'r')['data']
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = local_data
             
        return global_data, local_data, entry["properties"]
