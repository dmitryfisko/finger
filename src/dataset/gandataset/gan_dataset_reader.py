from pathlib import Path

import numpy as np
import pandas
from skimage.io import imread
from skimage.transform import resize


class GANDatasetReader:

    def read_dataset_item(self, file_path, target_size):
        image = self._read_normalized_tensor(file_path, target_size)
        keypoints = self._get_keypoints_normalized(file_path, target_size)
        return image, keypoints

    @staticmethod
    def _read_normalized_tensor(file_path, target_size):
        return resize(imread(file_path), (target_size, target_size), preserve_range=True).astype('uint8')

    def _get_keypoints_normalized(self, file_path, target_size):
        return self._get_keypoints(file_path) / 256 * target_size

    def _get_keypoints(self, file_path):
        file = Path(file_path)
        points2d_filename = file.name.split("_")[0] + "_joint2D.txt"
        points2d_file = Path(file.parent, points2d_filename)
        return self._read_tensor(points2d_file, (21, 2))
        # return self._read_tensor(points2d_file, (21, 2))[8].reshape((1, 2))

    @staticmethod
    def _read_tensor(file_path: str, shape):
        matrix = pandas.read_csv(file_path, header=None, dtype=np.float32).values
        matrix_numpy = np.resize(matrix, shape)
        return np.flip(matrix_numpy, axis=1).copy()
