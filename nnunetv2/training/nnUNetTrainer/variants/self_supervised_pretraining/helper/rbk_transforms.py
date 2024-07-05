from pathlib import Path

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

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

        data = data_dict["data"]
        batch_size = data.shape[0]

        result_dict = {"data": [], "order_label": [], "hor_label": [], "ver_label": []}

        for b in range(batch_size):
            d, o, h, v = self.rbk_transform(data[b])
            result_dict["data"].append(d)
            result_dict["order_label"].append(o)
            result_dict["hor_label"].append(h)
            result_dict["ver_label"].append(v)

        # make lists to array -> restore batch dimension
        for k in ["data", "order_label", "hor_label", "ver_label"]:
            result_dict[k] = np.stack(result_dict[k], axis=0)

        tensor_dict = NumpyToTensor(keys=["data", "order_label", "hor_label", "ver_label"])(**result_dict)
        return {"data": tensor_dict["data"],
                "target": {"order_label": tensor_dict["order_label"], "hor_label": tensor_dict["hor_label"],
                           "ver_label": tensor_dict["ver_label"]}}

    def rbk_transform(self, data):
        """
        Transforms the data for the RBK task.
        :param data: numpy ndarray of shape [C, H, W, D]
        :return dictionary containing the transformed data with labels
        """
        # crop the image to the same size
        cropped_data = self._crop_data(data, crop_size=self.crop_size)

        # extract cubes from 3d volume
        all_cubes = self._extract_3d_cubes(cropped_data)

        # task 1: rearrange cubes
        rearranged_cubes, order_label = self._rearrange_cubes(all_cubes)

        # task 2: rotate each cube randomly and return one-hot encoded labels
        rearranged_rotated_cubes, hor_label, ver_label = self._rotate_cubes(rearranged_cubes)

        return (
            rearranged_rotated_cubes,
            order_label,
            hor_label,
            ver_label,
        )

    @staticmethod
    def _crop_data(data, crop_size, center_crop=True):
        _, data_h, data_w, data_d = data.shape

        if center_crop:
            x_start = (data_h - crop_size[0]) // 2
            y_start = (data_w - crop_size[1]) // 2
            z_start = (data_d - crop_size[2]) // 2
        else:
            x_start = np.random.randint(0, data_h - crop_size[0])
            y_start = np.random.randint(0, data_w - crop_size[1])
            z_start = np.random.randint(0, data_d - crop_size[2])

        x_end = x_start + crop_size[0]
        y_end = y_start + crop_size[1]
        z_end = z_start + crop_size[2]

        return data[:, x_start:x_end, y_start:y_end, z_start:z_end]

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

                    p = data[
                        :,
                        i: i + h_grid + patch_overlap,
                        j: j + w_grid + patch_overlap,
                        k: k + d_grid + patch_overlap
                        ]

                    # crop the patch if the patch is smaller than the grid
                    if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                        p = self._crop_data(p, (h_patch, w_patch, d_patch))

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
