import copy

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnunetv2.training.nnUNetTrainer.variants.self_supervised_pretraining.helper.transform_utils import crop_data, \
    bezier_curve


class MGTransforms(AbstractTransform):
    def __init__(self, crop_size, flip_rate: float = 0.4, local_pixel_shuffling_rate: float = 0.5,
                 nonlinear_transformation_rate: float = 0.9, paint_rate: float = 0.9, inpainting_rate: float = 0.2):
        self.crop_size = crop_size
        self.flip_rate = flip_rate
        self.local_pixel_shuffling_rate = local_pixel_shuffling_rate
        self.nonlinear_transformation_rate = nonlinear_transformation_rate
        self.paint_rate = paint_rate
        self.inpainting_rate = inpainting_rate

    def __call__(self, **data_dict):
        data = data_dict["image"]

        # TODO: make this compatible with torch.Tensor
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        d, t = self.mg_transform(data)
        result_dict = {"image": d.astype(np.float32), "segmentation": t.astype(np.float32)}

        return NumpyToTensor(keys=["image", "segmentation"])(**result_dict)

    def mg_transform(self, data):
        """
        Transforms the data for the Models Genesis Task.
        :param data: numpy ndarray of shape [C, H, W, D]
        :return dictionary containing the transformed data
        This includes random cropping, pixel shuffling, nonlinear transformation and inpainting/outpainting.
        """
        input_patch = crop_data(data, crop_size=self.crop_size, center_crop=False)
        gt = copy.deepcopy(input_patch)

        # Flipping
        flipped_input_patch, flipped_gt = self._flip_image(input_patch, gt)

        # Local Shuffling Pixel
        shuffled_input_patch = self._local_pixel_shuffling(flipped_input_patch)

        # Apply non-Linear transformation with an assigned probability
        transformed_input_patch = self._nonlinear_transformation(shuffled_input_patch)

        # Inpainting & Outpainting
        if np.random.uniform() < self.paint_rate:
            painted_input_patch = self._inpainting(transformed_input_patch)
        else:
            painted_input_patch = self._outpainting(transformed_input_patch)

        from batchviewer import view_batch
        view_batch(painted_input_patch, width=300, height=300)

        return painted_input_patch, flipped_gt

    def _flip_image(self, ipt, gt):
        count = 3
        while np.random.uniform() < self.flip_rate and count > 0:
            axis = np.random.randint(0, 3)
            ipt = np.flip(ipt, axis=axis)
            gt = np.flip(gt, axis=axis)
            count -= 1

        return ipt, gt

    def _local_pixel_shuffling(self, ipt):
        if np.random.uniform() >= self.local_pixel_shuffling_rate:
            return ipt

        local_shuffling_x = copy.deepcopy(ipt)
        orig_image = copy.deepcopy(ipt)
        modalities, img_rows, img_cols, img_deps = ipt.shape
        num_block = 10000
        for _ in range(num_block):
            block_noise_size_x = np.random.randint(1, img_rows // 10)
            block_noise_size_y = np.random.randint(1, img_cols // 10)
            block_noise_size_z = np.random.randint(1, img_deps // 10)
            noise_x = np.random.randint(0, img_rows - block_noise_size_x)
            noise_y = np.random.randint(0, img_cols - block_noise_size_y)
            noise_z = np.random.randint(0, img_deps - block_noise_size_z)
            window = orig_image[:, noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y,
                     noise_z:noise_z + block_noise_size_z,
                     ]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((modalities, block_noise_size_x,
                                     block_noise_size_y,
                                     block_noise_size_z))
            local_shuffling_x[:, noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = window

        return local_shuffling_x

    def _nonlinear_transformation(self, x):
        if np.random.uniform() >= self.nonlinear_transformation_rate:
            return x
        points = [[0, 0], [np.random.uniform(), np.random.uniform()], [np.random.uniform(), np.random.uniform()],
                  [1, 1]]
        xvals, yvals = bezier_curve(points, nTimes=100000)
        if np.random.uniform() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    @staticmethod
    def _inpainting(x):
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 5
        while cnt > 0 and np.random.uniform() < 0.95:
            block_noise_size_x = np.random.randint(img_rows // 6, img_rows // 3)
            block_noise_size_y = np.random.randint(img_cols // 6, img_cols // 3)
            block_noise_size_z = np.random.randint(img_deps // 6, img_deps // 3)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x

    @staticmethod
    def _outpainting(x):
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - np.random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - np.random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - np.random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt = 4
        while cnt > 0 and np.random.uniform() < 0.95:
            block_noise_size_x = img_rows - np.random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - np.random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - np.random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y,
                                                    noise_z:noise_z + block_noise_size_z]
            cnt -= 1
        return x
