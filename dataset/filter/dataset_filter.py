import os
import random
from pathlib import Path
from distutils.dir_util import copy_tree


def get_all_dataset_dirs(dir):
    images = list()
    for root, dirs, files in os.walk(dir):
        if is_dataset_dirs(dirs):
            images += [os.path.join(root, directory) for directory in dirs]

    return images


def is_dataset_dirs(directories):
    return all([directory.isdigit() for directory in directories])


def select_percentage_of_list(array, percentage):
    random.shuffle(array)
    last_index = int(len(array) * percentage)
    return array[:last_index]


def make_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_dirs_to_output_dir(directories, output_directory):
    make_directory_exists(output_directory)
    for directory in directories:
        destination_folder_path = Path(directory)
        destination_folder_name = destination_folder_path.parent.name + '_' + destination_folder_path.name
        destination_directory = os.path.join(output_directory, destination_folder_name)
        copy_tree(directory, destination_directory)


if __name__ == '__main__':
    dataset_dirs = get_all_dataset_dirs('/Users/fisko/hands/regression/data/GANeratedDataset')
    selected_dataset_dirs = select_percentage_of_list(dataset_dirs, 0.2)
    copy_dirs_to_output_dir(selected_dataset_dirs, '/Users/fisko/hands/regression/data/GANeratedDatasetSelected')
