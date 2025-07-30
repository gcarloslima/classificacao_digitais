"""
Módulo de utilitários para preprocessamento e carregamento do modelo.

Este módulo contém funções auxiliares para:
- Preprocessamento de imagens para inferência
- Carregamento do modelo treinado
- Transformações de imagem padronizadas

As funções aqui são utilizadas tanto durante o treinamento quanto
na inferência através da interface Gradio.
"""

import os

import torch
import torchvision.transforms as transforms

from config import BEST_MODEL_PATH, DEVICE, IMAGE_SIZE

# Transformações padrão aplicadas nas imagens
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


def preprocess_image(image):
    """
    Preprocessa uma imagem PIL para inferência do modelo.

    Aplica as transformações necessárias (redimensionamento e conversão
    para tensor) e adiciona a dimensão do batch para compatibilidade
    com o modelo.

    Args:
        image (PIL.Image): Imagem a ser preprocessada

    Returns:
        torch.Tensor: Tensor preprocessado com shape (1, 3, height, width)
    """
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def load_model(num_classes):
    """
    Carrega o modelo treinado a partir do arquivo salvo.

    Cria uma instância do modelo, carrega os pesos salvos do melhor
    modelo durante o treinamento e configura para modo de avaliação.

    Args:
        num_classes (int): Número de classes para o modelo

    Returns:
        torch.nn.Module: Modelo carregado e pronto para inferência

    Raises:
        FileNotFoundError: Se o arquivo do modelo não for encontrado
    """
    #pylint: disable=wrong-import-position
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
