import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# set numpy seed for reproducibility
np.random.seed(42)


def convert_kits2023(cases_to_process: List[str], kits_base_dir: str, nnunet_dataset_id: int = 220):
    task_name = "KiTS2023"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    for tr in cases_to_process:
        shutil.copy(join(kits_base_dir, tr, 'imaging.nii.gz'), join(imagestr, f'{tr}_0000.nii.gz'))
        shutil.copy(join(kits_base_dir, tr, 'segmentation.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "kidney": (1, 2, 3),
                              "masses": (2, 3),
                              "tumor": 2
                          },
                          regions_class_order=(1, 3, 2),
                          num_training_cases=len(cases_to_process), file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='0.1.3',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="KiTS2023")


def split_and_preprocess(kits_data_dir: str, pretrain_id: int, finetune_id: int, pretrain_ratio: float):
    cases = subdirs(kits_data_dir, prefix='case_', join=False)
    num_cases = len(cases)
    num_split_cases = int(num_cases * pretrain_ratio)
    np.random.shuffle(cases)
    pretrain_cases = cases[:num_split_cases]
    finetune_cases = cases[num_split_cases:]

    convert_kits2023(pretrain_cases, kits_data_dir, pretrain_id)
    convert_kits2023(finetune_cases, kits_data_dir, finetune_id)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted KiTS2023 dataset (must have case_XXXXX subfolders)")
    parser.add_argument('-p', required=False, type=int, default=520,
                        help='nnU-Net Dataset ID for pretraining split, default: 520')
    parser.add_argument('-f', required=False, type=int, default=521,
                        help='nnU-Net Dataset ID for finetuning split, default: 512')
    parser.add_argument('-r', required=False, type=float, default=0.8, help='Pretraining ratio, default: 0.8')
    args = parser.parse_args()
    kits_base = args.input_folder
    split_and_preprocess(kits_base, args.p, args.f, args.r)
