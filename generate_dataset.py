from src.dataset.gandataset.dataset_tfrecords import create_tfrecords_dataset, count_items_in_tfrecord

DATASET_DIRECTORY = "/home/dmitryfisko/dataset/gan/GANeratedHands_Release/data"
TRAIN_FILE_PATH_TFRECORDS = "ganerated_256_train.tfrecords"
TEST_FILE_PATH_TFRECORDS = "ganerated_256_test.tfrecords"
TARGET_SIZE = 256

use_multiprocessing = False
use_compression = True
create_tfrecords_dataset(TARGET_SIZE, DATASET_DIRECTORY, TRAIN_FILE_PATH_TFRECORDS, TEST_FILE_PATH_TFRECORDS, 0.9,
                             use_compression=use_compression, use_multiprocessing=use_multiprocessing)
print(count_items_in_tfrecord(TRAIN_FILE_PATH_TFRECORDS, use_compression=use_compression))
print(count_items_in_tfrecord(TEST_FILE_PATH_TFRECORDS, use_compression=use_compression))