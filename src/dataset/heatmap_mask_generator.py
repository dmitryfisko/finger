import matplotlib.pyplot as plt
import numpy as np

from src.dataset.gandataset.gan_dataset_reader import GANDatasetReader


class HeatmapMaskGenerator:

    @staticmethod
    def _gaussian_k(x0, y0, sigma, width, height):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def generate_heatmap(self, height, width, landmark, sigma=7):
        return self._gaussian_k(landmark[1], landmark[0], sigma, height, width)


def visualuze_all_hand_points(height, width, keypoints):
    result_mask = np.zeros((height, width), dtype=np.float64)
    for i in range(len(keypoints)):
        mask = mask_generator.generate_heatmap(height, width, keypoints[8])
        result_mask += mask
    return result_mask


def plot_image_and_mask(image, mask):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.set_title("image")
    ax = fig.add_subplot(1, 2, 2)
    mask_width, mask_height, _ = mask.shape
    mask = mask.reshape((mask_width, mask_height))
    ax.imshow(mask, cmap="gray")
    ax.set_title("mask")
    plt.show()


if __name__ == '__main__':
    reader = GANDatasetReader()
    mask_generator = HeatmapMaskGenerator()

    image, keypoints = reader.read_dataset_item(
        '/Users/fisko/master/finger/data/GANeratedDatasetSelected/noObject/0003_color_composed.png', 256)

    height, width, _ = image.shape
    mask = mask_generator.generate_heatmap(height, width, keypoints[8])
    plot_image_and_mask(image, mask)
