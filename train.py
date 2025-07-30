"""
Módulo principal de treinamento do modelo de classificação de impressões digitais.

Este módulo contém toda a lógica de treinamento, incluindo:
- Carregamento e preparação dos dados
- Configuração do modelo e otimizador
- Loop de treinamento com validação
- Cálculo de métricas de avaliação
- Salvamento dos melhores modelos
- Geração de logs e gráficos de perda

O treinamento utiliza validação cruzada para monitorar o desempenho
e salva automaticamente o melhor modelo baseado no F1-score.
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from colorlog import ColoredFormatter
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, BEST_MODEL_PATH, DEVICE, IMAGE_SIZE,
                    LEARNING_RATE, MODEL_PATH, NUM_EPOCHS, TRAIN_DIR, VAL_DIR)
from dataset import FingerprintDataset
from model import SimpleFingerprintClassifier

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
    """
    Cria e retorna os DataLoaders para treinamento e validação.
    
    Aplica transformações apropriadas nas imagens (redimensionamento e
    conversão para tensor) e cria os DataLoaders com o tamanho de batch
    especificado nas configurações.
    
    Returns:
        tuple: (train_loader, val_loader, class_names) onde:
            - train_loader: DataLoader para dados de treinamento
            - val_loader: DataLoader para dados de validação  
            - class_names: Lista com nomes das classes
    """
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
    """
    Calcula métricas de avaliação para classificação.
    
    Args:
        y_true (array-like): Rótulos verdadeiros
        y_pred (array-like): Predições do modelo
        
    Returns:
        dict: Dicionário contendo F1-score, precisão e recall médios ponderados
    """
    return {
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def train_model():
    """
    Função principal de treinamento do modelo.
    
    Executa o loop completo de treinamento incluindo:
    - Inicialização do modelo, otimizador e função de perda
    - Treinamento por épocas com validação
    - Monitoramento de métricas e salvamento do melhor modelo
    - Geração de logs detalhados e gráfico de perda
    
    Returns:
        tuple: (modelo_treinado, nomes_das_classes)
    """
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
        logging.info("Epoch %d/%d", epoch + 1, NUM_EPOCHS)
        model.train()
        total_train_loss = 0

        # Loop de treinamento
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

        # Loop de validação
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

        logging.info("Train Loss: %.4f | Val Loss: %.4f", train_loss, val_loss)
        logging.info(
            "F1 Score: %.4f | Precision: %.4f | Recall: %.4f",
            metrics['f1_score'], metrics['precision'], metrics['recall']
        )
        
        # Salva o melhor modelo baseado no F1-score
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            os.makedirs("saved_models", exist_ok=True)
            torch.save(best_model_state, BEST_MODEL_PATH)
            logging.info("Novo melhor modelo salvo (Epoch %d) com F1 Score: %.4f", best_epoch, best_f1)

    torch.save(model.state_dict(), MODEL_PATH)
    logging.info("Último modelo salvo em %s", MODEL_PATH)
    logging.info("Melhor modelo salvo em %s na época %d com F1 Score: %.4f", BEST_MODEL_PATH, best_epoch, best_f1)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig("logs/loss_plot.png")
    logging.info("Gráfico de perda salvo em logs/loss_plot.png")

    return model, class_names
