# train.py
import os
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from config import *
from dataset import FingerprintDataset
from model import SimpleFingerprintClassifier
from colorlog import ColoredFormatter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'
)

color_formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(color_formatter)
logging.getLogger('').addHandler(console)


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


def compute_metrics(y_true, y_pred):
    return {
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        # "report": classification_report(y_true, y_pred, zero_division=0)  # opcional
    }


def train_model():
    logging.info("Iniciando treinamento...")
    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = SimpleFingerprintClassifier(num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_f1 = 0.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        metrics = compute_metrics(all_labels, all_preds)

        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logging.info(
            f"F1 Score: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )

        # logging.info(f"Classification Report:\n{metrics['report']}")  # se quiser ver o detalhado por classe

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            os.makedirs("saved_models", exist_ok=True)
            torch.save(best_model_state, BEST_MODEL_PATH)
            logging.info(f"Novo melhor modelo salvo (Epoch {best_epoch}) com F1 Score: {best_f1:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Último modelo salvo em {MODEL_PATH}")
    logging.info(f"Melhor modelo salvo em {BEST_MODEL_PATH} na época {best_epoch} com F1 Score: {best_f1:.4f}")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig("logs/loss_plot.png")
    logging.info("Gráfico de perda salvo em logs/loss_plot.png")

    return model, class_names
