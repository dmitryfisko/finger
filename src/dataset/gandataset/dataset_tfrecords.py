import random

import tensorflow as tf
from tensorflow_core.python.lib.io.tf_record import TFRecordWriter, tf_record_iterator

from src.dataset.gandataset.dataset_utils import find_all_dataset_images
from src.dataset.gandataset.gan_dataset_reader import GANDatasetReader
from src.dataset.heatmap_mask_generator import HeatmapMaskGenerator, plot_image_and_mask

DATASET_DIRECTORY = "/Users/fisko/master/finger/data/GANeratedDatasetSelected"
TRAIN_FILE_PATH_TFRECORDS = "/Users/fisko/master/finger/data/train.tfrecords"
TEST_FILE_PATH_TFRECORDS = "/Users/fisko/master/finger/data/test.tfrecords"
TARGET_SIZE = 256


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))


def create_tfrecords_dataset(
        target_size, dataset_directory, train_file_path_tfrecords, test_file_path_tfrecords, train_percent):
    dataset_image_pathes = find_all_dataset_images(dataset_directory)
    train_writer = TFRecordWriter(train_file_path_tfrecords)
    test_writer = TFRecordWriter(test_file_path_tfrecords)

    reader = GANDatasetReader()
    mask_generator = HeatmapMaskGenerator()

    random.seed(42)
    for i, dataset_image_path in enumerate(dataset_image_pathes):
        image, keypoints = reader.read_dataset_item(dataset_image_path, target_size)
        height, width, _ = image.shape
        mask = mask_generator.generate_heatmap(height, width, keypoints[8])

        features = {
            'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
            'mask': _bytes_feature(tf.compat.as_bytes(mask.tostring()))
        }
        item = tf.train.Example(features=tf.train.Features(feature=features))

        serialized_item = item.SerializeToString()
        if random.uniform(0, 1) <= train_percent:
            train_writer.write(serialized_item)
        else:
            test_writer.write(serialized_item)

        if i % 1000 == 0:
            print('iteration:', i)

    train_writer.close()
    test_writer.close()


def parse_tfrecord_function(proto):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.io.parse_single_example(proto, features)
    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    mask = tf.io.decode_raw(parsed_features['mask'], tf.float64)

    return tf.reshape(image, (TARGET_SIZE, TARGET_SIZE, 3)), tf.reshape(mask, (TARGET_SIZE, TARGET_SIZE, 1))


def count_items_in_tfrecord(file_path_tfrecord):
    line_count = 0
    for dataset_item in tf_record_iterator(file_path_tfrecord):
        # image, mask = parse_tfrecord_function(dataset_item)
        # plot_image_and_mask(image.numpy(), mask.numpy())
        line_count += 1

    return line_count


if __name__ == "__main__":
    create_tfrecords_dataset(TARGET_SIZE, DATASET_DIRECTORY, TRAIN_FILE_PATH_TFRECORDS, TEST_FILE_PATH_TFRECORDS, 0.5)
    print(count_items_in_tfrecord(TRAIN_FILE_PATH_TFRECORDS))
    print(count_items_in_tfrecord(TEST_FILE_PATH_TFRECORDS))
