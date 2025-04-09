import os

BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
TRAIN_FILE = os.path.join(DATASET_PATH, 'train.csv')
TEST_FILE = os.path.join(DATASET_PATH, 'test.csv')
