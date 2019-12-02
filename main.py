from dataset.reader.gan_dataset import GANDatasetReader
from dataset.heatmap_mask_generator import HeatmapMaskGenerator
import matplotlib.pyplot as plt
import numpy as np


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
    ax.imshow(mask, cmap="gray")
    ax.set_title("mask")
    plt.show()


if __name__ == '__main__':
    reader = GANDatasetReader()
    mask_generator = HeatmapMaskGenerator()

    image, keypoints = reader.read_dataset_item(
        '/Users/fisko/master/finger/data/GANeratedDatasetSelected/noObject_0003/0004_color_composed.png', 256)

    height, width, _ = image.shape
    result_mask = np.zeros((height, width), dtype=np.float64)
    mask = mask_generator.generate_heatmap(height, width, keypoints[8])
    plot_image_and_mask(image, mask)
