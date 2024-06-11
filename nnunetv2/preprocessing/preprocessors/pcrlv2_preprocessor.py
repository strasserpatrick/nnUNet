from dataclasses import dataclass
from typing import Union

import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from skimage.transform import resize


@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing
    :param input_rows: int, size of global window in x-axis
    :param input_cols: int, size of global window in y-axis
    :param input_depth: int, size of global window in z axis
    :param scale: int, number of 3D pairs to generate
    :param local_input_rows: int, size of local window in x-axis
    :param local_input_cols: int, size of local window in y-axis
    :param local_input_depth: int, size of local window in z-axis
    :param len_border: int, length of minimal distance to image boarder in x- and y-axis when cropping global windows
    :param len_border_z: int, length of minimal distance to image boarder in z-axis when cropping global windows
    :param len_depth: int, z dimensional extension of the global window
    :param col_size_sampling_variants: list, possible sizes for global windows
    :param local_col_size_sampling_variants: list, possible sizes for local windows
    """
    input_rows: int
    input_cols: int
    input_depth: int
    scale: float
    local_input_rows: int = 16
    local_input_cols: int = 16
    local_input_depth: int = 16
    len_border: int = 70
    len_border_z: int = 15
    len_depth: int = 3
    lung_max: float = 0.15
    col_size_sampling_variants = [(96, 96, 64), (96, 96, 96), (112, 112, 64), (64, 64, 32)]
    local_col_size_sampling_variants = [(32, 32, 16), (16, 16, 16), (32, 32, 32), (8, 8, 8)]


class PCRLv2Preprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)

        self.config = PreprocessingConfig(input_rows=256, input_cols=256, input_depth=256, scale=6)

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        data, _ = super().run_case_npy(data, seg, properties, plans_manager, configuration_manager, dataset_json)

        crop_windows = []
        for i in range(self.config.scale):
            crop_window1, crop_window2, local_windows = self.crop_pair(data)
            crop_window = np.stack((crop_window1, crop_window2), axis=0)
            crop_windows.append(crop_window)

        return crop_windows, None

    def crop_pair(self, img_array):
        while True:
            # TODO: we have four modalities, how to handle them? is this really worth the effort exploring?
            size_x, size_y, size_z = img_array.shape
            img_array1 = img_array.copy()
            img_array2 = img_array.copy()

            # padding the both images on the z axis, if image is too small
            if size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z < self.config.len_border_z:
                pad_array1 = size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z - self.config.len_border_z
                padding_array1 = [0, 0, -pad_array1 + 1]
                img_array1 = np.pad(img_array1, padding_array1, mode='constant', constant_values=0)

                pad_array2 = size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z - self.config.len_border_z
                padding_array2 = [0, 0, -pad_array2 + 1]
                img_array2 = np.pad(img_array2, padding_array2, mode='constant', constant_values=0)
                size_z += -pad_array2 + 1

            # sample the global windows and unpack the boxes
            box_1, box_2, crops = self._sample_global_windows((size_x, size_y, size_z))
            start_x1, end_x1, start_y1, end_y1, start_z1, end_z1 = box_1
            start_x2, end_x2, start_y2, end_y2, start_z2, end_z2 = box_2
            crop_rows1, crop_cols1, crop_deps1, crop_rows2, crop_cols2, crop_deps2 = crops

            # crop global view windows out of images
            crop_window1 = img_array1[start_x1: start_x1 + crop_rows1,
                           start_y1: start_y1 + crop_cols1,
                           start_z1: start_z1 + crop_deps1 + self.config.len_depth,
                           ]

            crop_window2 = img_array2[start_x2: start_x2 + crop_rows2,
                           start_y2: start_y2 + crop_cols2,
                           start_z2: start_z2 + crop_deps2 + self.config.len_depth,
                           ]

            # resize crop windows to match self.config.input rows, cols and depth requirements
            if crop_rows1 != self.config.input_rows or crop_cols1 != self.config.input_cols or crop_deps1 != self.config.input_depth:
                crop_window1 = resize(crop_window1,
                                      (self.config.input_rows, self.config.input_cols,
                                       self.config.input_depth + self.config.len_depth),
                                      preserve_range=True,
                                      )
            if crop_rows2 != self.config.input_rows or crop_cols2 != self.config.input_cols or crop_deps2 != self.config.input_depth:
                crop_window2 = resize(crop_window2,
                                      (self.config.input_rows, self.config.input_cols,
                                       self.config.input_depth + self.config.len_depth),
                                      preserve_range=True,
                                      )
            t_img1 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            d_img1 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            t_img2 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            d_img2 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            for d in range(self.config.input_depth):
                for i in range(self.config.input_rows):
                    for j in range(self.config.input_cols):
                        for k in range(self.config.len_depth):
                            if crop_window1[
                                i, j, d + k] >= self.config.HU_thred:  # TODO: what does that mean? how to set it for z-normalization?
                                t_img1[i, j, d] = crop_window1[i, j, d + k]
                                d_img1[i, j, d] = k
                                break
                            if k == self.config.len_depth - 1:
                                d_img1[i, j, d] = k
            for d in range(self.config.input_depth):
                for i in range(self.config.input_rows):
                    for j in range(self.config.input_cols):
                        for k in range(self.config.len_depth):
                            if crop_window2[i, j, d + k] >= self.config.HU_thred:
                                t_img2[i, j, d] = crop_window2[i, j, d + k]
                                d_img2[i, j, d] = k
                                break
                            if k == self.config.len_depth - 1:
                                d_img2[i, j, d] = k

            d_img1 = d_img1.astype('float32')
            d_img1 /= (self.config.len_depth - 1)
            d_img1 = 1.0 - d_img1
            d_img2 = d_img2.astype('float32')
            d_img2 /= (self.config.len_depth - 1)
            d_img2 = 1.0 - d_img2

            if np.sum(d_img1) > self.config.lung_max * crop_cols1 * crop_deps1 * crop_rows1:
                continue
            if np.sum(d_img2) > self.config.lung_max * crop_cols1 * crop_deps1 * crop_rows1:
                continue
            # we start to crop the local windows
            x_min = min(start_x1, start_x2)
            x_max = max(end_x1, end_x2)
            y_min = min(start_y1, start_y2)
            y_max = max(end_y1, end_y2)
            z_min = min(start_z1, start_z2)
            z_max = max(end_z1, end_z2)
            local_windows = []
            for i in range(6):
                local_x = np.random.randint(max(x_min - 3, 0), min(x_max + 3, size_x))
                local_y = np.random.randint(max(y_min - 3, 0), min(y_max + 3, size_y))
                local_z = np.random.randint(max(z_min - 3, 0), min(z_max + 3, size_z))
                local_size_index = np.random.randint(0, len(self.config.local_col_size_sampling_variants))
                local_crop_rows, local_crop_cols, local_crop_deps = self.config.local_col_size_sampling_variants[
                    local_size_index]
                local_window = img_array1[local_x: local_x + local_crop_rows,
                               local_y: local_y + local_crop_cols,
                               local_z: local_z + local_crop_deps
                               ]
                # if local_crop_rows != local_input_rows or local_crop_cols != local_input_cols or local_crop_deps != local_input_depth:
                local_window = resize(local_window,
                                      (self.config.local_input_rows, self.config.local_input_cols,
                                       self.config.local_input_depth),
                                      preserve_range=True,
                                      )
                local_windows.append(local_window)
            return crop_window1[:, :, :self.config.input_depth], crop_window2[:, :, :self.config.input_depth], np.stack(
                local_windows, axis=0)

    def _sample_global_windows(self, image_sizes, min_iou: float = 0.3):
        """
        Samples two global windows from the image

        :param image_sizes: tuple, size of the image
        :param min_iou: float, minimum intersection over union value

        :return: tuple, coordinates of the two windows and crop parameters
        """
        size_x, size_y, size_z = image_sizes

        while True:

            # randomly sample the size of the crop window out of the variants
            size_index1 = np.random.randint(0, len(self.config.col_size_sampling_variants))
            crop_rows1, crop_cols1, crop_deps1 = self.config.col_size_sampling_variants[size_index1]
            size_index2 = np.random.randint(0, len(self.config.col_size_sampling_variants))
            crop_rows2, crop_cols2, crop_deps2 = self.config.col_size_sampling_variants[size_index2]

            # reduce the size of the crop window if it is too close to the border
            if size_x - crop_rows1 - 1 - self.config.len_border <= self.config.len_border:
                crop_rows1 -= 32
                crop_cols1 -= 32
            if size_x - crop_rows2 - 1 - self.config.len_border <= self.config.len_border:
                crop_rows2 -= 32
                crop_cols2 -= 32

            # sample the box start coordinates of the crop windows
            # the end coordinates are start + crop param
            # a minimum distance to the boarder is enforced by the len_border parameter
            # additionally, for the z axis, a minimum distance to the top and bottom is enforced by len_border_z
            start_x1 = np.random.randint(0 + self.config.len_border,
                                         size_x - crop_rows1 - 1 - self.config.len_border)
            start_y1 = np.random.randint(0 + self.config.len_border,
                                         size_y - crop_cols1 - 1 - self.config.len_border)
            start_z1 = np.random.randint(0 + self.config.len_border_z,
                                         size_z - crop_deps1 - self.config.len_depth - 1 - self.config.len_border_z)
            start_x2 = np.random.randint(0 + self.config.len_border,
                                         size_x - crop_rows2 - 1 - self.config.len_border)
            start_y2 = np.random.randint(0 + self.config.len_border,
                                         size_y - crop_cols2 - 1 - self.config.len_border)
            start_z2 = np.random.randint(0 + self.config.len_border_z,
                                         size_z - crop_deps2 - self.config.len_depth - 1 - self.config.len_border_z)
            box1 = (
                start_x1, start_x1 + crop_rows1, start_y1, start_y1 + crop_cols1, start_z1, start_z1 + crop_deps1)
            box2 = (
                start_x2, start_x2 + crop_rows2, start_y2, start_y2 + crop_cols2, start_z2, start_z2 + crop_deps2)
            iou = self._calculate_iou(box1, box2)

            if iou > min_iou:
                crops = (crop_rows1, crop_cols1, crop_deps1, crop_rows2, crop_cols2, crop_deps2)
                return box1, box2, crops

    @staticmethod
    def _calculate_iou(box1, box2):
        """
        Calculates Intersection over Union for two boxes

        :param box1: tuple, coordinates of the first box
        :param box2: tuple, coordinates of the second box

        :return: float, Intersection over Union value
        """

        # unpack the coordinates from boxes
        x_start_box1, x_end_box1, y_start_box1, y_end_box1, z_start_box1, z_end_box1 = box1
        x_start_box2, x_end_box2, y_start_box2, y_end_box2, z_start_box2, z_end_box2 = box2

        # compute the volume of boxes
        area_box1 = (x_end_box1 - x_start_box1) * (y_end_box1 - y_start_box1) * (z_end_box1 - z_start_box1)
        area_box2 = (x_end_box2 - x_start_box2) * (y_end_box2 - y_start_box2) * (z_end_box2 - z_start_box2)

        # find the intersection box and compute the volume
        x_min = max(x_start_box1, x_start_box2)
        y_min = max(y_start_box1, y_start_box2)
        x_max = min(x_end_box1, x_end_box2)
        y_max = min(y_end_box1, y_end_box2)
        z_min = max(z_start_box1, z_start_box2)
        z_max = min(z_end_box1, z_end_box2)

        intersection_w = max(0, x_max - x_min)
        intersection_h = max(0, y_max - y_min)
        intersection_d = max(0, z_max - z_min)
        intersection_area = intersection_w * intersection_h * intersection_d

        # compute the intersection over union
        iou = intersection_area / (area_box1 + area_box2 - intersection_area)
        return iou


def example_test_case_preprocessing():
    plans_file = "/Users/patricks/Workspace/uni/LS6/masterarbeit/data/nnUNet_preprocessed/Dataset300_BraTS2021_pretraining/nnUNetPreprocessPlans.json"
    dataset_json_file = "/Users/patricks/Workspace/uni/LS6/masterarbeit/data/nnUNet_preprocessed/Dataset300_BraTS2021_pretraining/dataset.json"
    input_images = [
        "/Users/patricks/Workspace/uni/LS6/masterarbeit/data/nnUNet_raw_data/Dataset300_BraTS2021_pretraining/imagesTr/BraTS2021_00234_0000.nii.gz",
        "/Users/patricks/Workspace/uni/LS6/masterarbeit/data/nnUNet_raw_data/Dataset300_BraTS2021_pretraining/imagesTr/BraTS2021_00234_0001.nii.gz",
        "/Users/patricks/Workspace/uni/LS6/masterarbeit/data/nnUNet_raw_data/Dataset300_BraTS2021_pretraining/imagesTr/BraTS2021_00234_0002.nii.gz",
        "/Users/patricks/Workspace/uni/LS6/masterarbeit/data/nnUNet_raw_data/Dataset300_BraTS2021_pretraining/imagesTr/BraTS2021_00234_0003.nii.gz",
    ]

    configuration = "3d_fullres"
    pp = PCRLv2Preprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(
        input_images,
        seg_file=None,
        plans_manager=plans_manager,
        configuration_manager=plans_manager.get_configuration(configuration),
        dataset_json=dataset_json_file,
    )

    return data


if __name__ == "__main__":
    example_test_case_preprocessing()
