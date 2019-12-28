from src.dataset.gandataset.dataset_tfrecords import count_items_in_tfrecord

DATASET_DIRECTORY = "/home/dmitryfisko/dataset/gan/GANeratedHands_Release/data"
TRAIN_FILE_PATH_TFRECORDS = "ganerated_256_train.tfrecords"
TEST_FILE_PATH_TFRECORDS = "ganerated_256_test.tfrecords"

use_multiprocessing = False
use_compression = False
print(count_items_in_tfrecord(TRAIN_FILE_PATH_TFRECORDS, use_compression=use_compression))
print(count_items_in_tfrecord(TEST_FILE_PATH_TFRECORDS, use_compression=use_compression))