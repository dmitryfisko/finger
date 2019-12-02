import os
import random


def get_train_and_validation_images(dir, train_percent):
    dataset_images = find_all_dataset_images(dir)

    return _split_array(dataset_images, train_percent)


def _split_array(array, percentage):
    size = int(round(percentage * len(array)))
    return array[:size], array[size:]


def find_all_dataset_images(directory, search_dirs=('noObject', )): #withObject
    images = list()
    for search_dir in search_dirs:
        images.extend(_find_all_folder_images(os.path.join(directory, search_dir)))

    if len(images) == 0:
        raise RuntimeError('Bad GANNerated dataset structure')

    # dirty hack with filtering joints.txt
    dataset_images = list(filter(lambda image_path: 'joint' not in image_path, images))
    random.Random(42).shuffle(dataset_images)
    return dataset_images


def _find_all_folder_images(dir):
    images = list()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".png"):
                images.append(os.path.join(root, file))

    return images
