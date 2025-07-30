
import torch
import os
from PIL import Image
from config import *
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def preprocess_image(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def load_model(num_classes):
    from model import SimpleFingerprintClassifier
    model = SimpleFingerprintClassifier(num_classes)
    
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"\n❌ Modelo não encontrado em '{BEST_MODEL_PATH}'. "
            "Execute 'python train_model.py' para treinar o modelo.\n"
        )
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model
