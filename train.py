import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from config import *
from dataset import FingerprintDataset
from model import SimpleFingerprintClassifier
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    train_set = FingerprintDataset(TRAIN_DIR, transform)
    val_set = FingerprintDataset(VAL_DIR, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, train_set.classes

def train_model():
    train_loader, val_loader, class_names = get_dataloaders()
    
    model = SimpleFingerprintClassifier(num_classes=len(class_names)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")
    
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")

    # Plot
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

    return model, class_names
