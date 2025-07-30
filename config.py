import torch

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 120
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'saved_models/fingerprint_model.pth'
BEST_MODEL_PATH = 'saved_models/fingerprint_model_best.pth'