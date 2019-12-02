from gan_dataset import GANDatasetReader
from heatmap_mask_generator import HeatmapMaskGenerator
import matplotlib.pyplot as plt


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
        '/Users/fisko/master/finger/data/GANeratedDatasetSelected/noObject_0003/0001_color_composed.png', 256)

    height, width, _ = image.shape
    mask = mask_generator.generate_heatmap(height, width, keypoints[0])
    plot_image_and_mask(image, mask)
