import torch

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 120
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'saved_models/fingerprint_model.pth'