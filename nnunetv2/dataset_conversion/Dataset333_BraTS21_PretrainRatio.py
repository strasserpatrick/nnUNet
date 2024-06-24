import multiprocessing
import shutil
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

file_dir = Path(__file__).parent


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str,
                                                                num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


def split_cases(case_ids: List[str], ratio: float):
    """
    Splits cases into two portions based on the given ratio.
    :param case_ids: List of case IDs.
    :param ratio: Ratio of cases to be included in the first portion.
    :return: Two portions of case IDs.
    """
    num_cases = len(case_ids)
    num_split_cases = int(num_cases * ratio)
    np.random.shuffle(case_ids)
    portion_a = case_ids[:num_split_cases]
    portion_b = case_ids[num_split_cases:]
    return portion_a, portion_b


def process_portion(brats_data_dir, out_foldername, case_ids):
    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, out_foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    for c in tqdm(case_ids):
        shutil.copy(join(brats_data_dir, c, c + "_t1.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_t1ce.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_t2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "_flair.nii.gz"), join(imagestr, c + '_0003.nii.gz'))

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "_seg.nii.gz"),
                                                             join(labelstr, c + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (2, 3),
                              'enhancing tumor': (3,)
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')


def preprocess_brats21(pretrain_ratio: float, task_id_pretrain=333, task_id_finetune=334, task_name_prefix="BraTS2021"):
    brats_data_dir = '/Users/patricks/Downloads/brats2021/BraTS2021_Training_Data'

    pretrain_task_name = f"{task_name_prefix}_{pretrain_ratio:.1f}".replace(".", "_")
    pretrain_foldername = f"Dataset{task_id_pretrain:03.0f}_{pretrain_task_name}"

    finetune_task_name = f"{task_name_prefix}_{1 - pretrain_ratio:.1f}".replace(".", "_")
    finetune_foldername = f"Dataset{task_id_finetune:03.0f}_{finetune_task_name}"

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)
    pretrain_case_ids, finetuning_case_ids = split_cases(case_ids, pretrain_ratio)

    print("Process pretrained cases")
    process_portion(brats_data_dir, pretrain_foldername, pretrain_case_ids)

    print("Process finetuning cases")
    process_portion(brats_data_dir, finetune_foldername, finetuning_case_ids)


if __name__ == '__main__':
    preprocess_brats21(pretrain_ratio=0.9)
