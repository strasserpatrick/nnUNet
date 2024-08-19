from pathlib import Path

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.transform_utils import crop_data

helper_dir = Path(__file__).parent


class RBKTransform(AbstractTransform):
    def __init__(self, crop_size, num_cubes_per_side=2, jitter_xy=10, jitter_z=5):
        self.crop_size = crop_size

        self.num_cubes_per_side = num_cubes_per_side
        self.num_cubes = num_cubes_per_side ** 3

        self.jitter_xy = jitter_xy
        self.jitter_z = jitter_z

        self.k_permutations = np.load(str(helper_dir / "k_permutations.npy"))

    def __call__(self, **data_dict):

        data = data_dict["image"]

        d, o, h, v = self.rbk_transform(data)
        result_dict = {"image": d, "order_label": o, "hor_label": h, "ver_label": v}

        tensor_dict = NumpyToTensor(keys=["image", "order_label", "hor_label", "ver_label"])(**result_dict)

        return {"image": tensor_dict["image"],
                "segmentation": {"order_label": tensor_dict["order_label"], "hor_label": tensor_dict["hor_label"],
                                 "ver_label": tensor_dict["ver_label"]}}

    def rbk_transform(self, data):
        """
        Transforms the data for the RBK task.
        :param data: numpy ndarray of shape [C, H, W, D]
        :return dictionary containing the transformed data with labels
        """
        # crop the image to the same size
        cropped_data = crop_data(data, crop_size=self.crop_size)

        # extract cubes from 3d volume
        all_cubes = self._extract_3d_cubes(cropped_data)

        # task 1: rearrange cubes
        rearranged_cubes, order_label = self._rearrange_cubes(all_cubes)

        # task 2: rotate each cube randomly and return one-hot encoded labels
        rearranged_rotated_cubes, hor_label, ver_label = self._rotate_cubes(rearranged_cubes)

        return (
            np.array(rearranged_rotated_cubes),
            np.array([order_label]),
            np.array(hor_label),
            np.array(ver_label),
        )

    def _extract_3d_cubes(self, data):
        """
        Crops cubes from 3D Image
        :param data: numpy ndarray
        :return: list of numpy ndarray
        """
        _, h, w, d = data.shape

        patch_overlap = -self.jitter_xy if self.jitter_xy < 0 else 0

        h_grid = (h - patch_overlap) // self.num_cubes_per_side
        w_grid = (w - patch_overlap) // self.num_cubes_per_side
        d_grid = (d - patch_overlap) // self.num_cubes_per_side
        h_patch = h_grid - self.jitter_xy
        w_patch = w_grid - self.jitter_xy
        d_patch = d_grid - self.jitter_z

        cubes = []
        for i in range(self.num_cubes_per_side):
            for j in range(self.num_cubes_per_side):
                for k in range(self.num_cubes_per_side):

                    start_h = i * h_grid
                    start_w = j * w_grid
                    start_d = k * d_grid

                    p = data[
                        :,
                        start_h:start_h + h_grid + patch_overlap,
                        start_w:start_w + w_grid + patch_overlap,
                        start_d:start_d + d_grid + patch_overlap
                        ]

                    # crop the patch if the patch is smaller than the grid
                    if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                        p = crop_data(p, (h_patch, w_patch, d_patch))

                    cubes.append(p)

        return cubes

    def _rearrange_cubes(self, cubes):
        """
        Rearrange the order of cubes.
        :param cubes: list of numpy ndarray
        :return: list of numpy ndarray and the index of permutation used (from K_permutations)
        """
        label = np.random.randint(0, len(self.k_permutations) - 1)
        return np.array(cubes)[np.array(self.k_permutations[label])], label

    def _rotate_cubes(self, cubes):
        """
        Rotate each cube randomly.
        With a probability of 1/3, rotate 180 along x-axis.
        With a probability of 1/3, rotate 180 along y-axis.
        With a probability of 1/3, do not rotate.
        :param cubes: list of numpy ndarray
        :return (rotated cubes, horizontal one-hot vector, vertical one-hot vector) one hot vector indicates the rotation
        """

        # multi-hot labels
        # [8, H, W, D]
        rot_cubes = cubes.copy()
        hor_vector = []
        ver_vector = []

        for i in range(self.num_cubes):
            p = np.random.random()
            cube = rot_cubes[i]
            # [H, W, D]
            if p < 1 / 3:
                hor_vector.append(1)
                ver_vector.append(0)
                # rotate 180 along x-axis
                rot_cubes[i] = np.flip(cube, (2, 3))
            elif p < 2 / 3:
                hor_vector.append(0)
                ver_vector.append(1)
                # rotate 180 along z-axis
                rot_cubes[i] = np.flip(cube, (1, 2))

            else:
                hor_vector.append(0)
                ver_vector.append(0)

        return rot_cubes, hor_vector, ver_vector
