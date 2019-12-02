import numpy as np


class HeatmapMaskGenerator:

    @staticmethod
    def _gaussian_k(x0, y0, sigma, width, height):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def generate_heatmap(self, height, width, landmark, sigma=3):
        return self._gaussian_k(landmark[0], landmark[1], sigma, height, width)
