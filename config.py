"""
Arquivo de configuração para o sistema de classificação de impressões digitais.

Este módulo contém todas as constantes e configurações utilizadas durante
o treinamento e inferência do modelo de classificação de impressões digitais.
"""

import torch

# Parâmetros de treinamento
BATCH_SIZE = 128  # Tamanho do lote para treinamento
NUM_EPOCHS = 30  # Número de épocas de treinamento
LEARNING_RATE = 0.001  # Taxa de aprendizagem para o otimizador
IMAGE_SIZE = (256, 256)  # Dimensões de redimensionamento das imagens
NUM_CLASSES = 120  # Número total de classes (pessoas)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Dispositivo para computação

# Caminhos dos diretórios de dados
TRAIN_DIR = "dataset/train"  # Diretório com dados de treinamento
VAL_DIR = "dataset/valid"  # Diretório com dados de validação
TEST_DIR = "dataset/test"  # Diretório com dados de teste

# Caminhos para salvamento dos modelos
MODEL_PATH = "saved_models/fingerprint_model.pth"  # Modelo final
BEST_MODEL_PATH = "saved_models/fingerprint_model_best.pth"  # Melhor modelo durante treinamento
