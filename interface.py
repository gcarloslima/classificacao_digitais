"""
Interface gráfica usando Gradio para classificação de impressões digitais.

Este módulo cria uma interface web interativa que permite aos usuários
carregar imagens de impressões digitais e receber predições do modelo treinado.
A interface exibe a imagem carregada e as probabilidades de classificação.
"""

import gradio as gr
import torch

from config import DEVICE, TRAIN_DIR
from dataset import FingerprintDataset
from utils import load_model, preprocess_image

# Carrega classes e modelo treinado
classes_dataset = FingerprintDataset(TRAIN_DIR)
class_names = classes_dataset.classes
model = load_model(len(class_names))


def classify(image):
    """
    Classifica uma imagem de impressão digital.

    Esta função recebe uma imagem PIL, a preprocessa e executa a inferência
    usando o modelo treinado. Retorna tanto a imagem original quanto as
    probabilidades de classificação para cada classe.

    Args:
        image (PIL.Image): Imagem de impressão digital a ser classificada

    Returns:
        tuple: (imagem_original, dicionário_de_confianças) onde o dicionário
               mapeia nomes de classes para suas respectivas probabilidades
    """
    tensor = preprocess_image(image)
    probabilities = model(tensor.to(DEVICE))
    probs = (
        torch.nn.functional.softmax(probabilities, dim=1)
        .cpu()
        .detach()
        .numpy()
        .flatten()
    )
    confidences = {cls: float(prob) for cls, prob in zip(class_names, probs)}

    # ALTERAÇÃO 1: Retornar a imagem original junto com as classificações.
    # A função agora retorna uma tupla: (imagem, dicionário_de_confianças)
    return image, confidences


# ALTERAÇÃO 2: Atualizar os outputs para receber uma imagem e um label.
interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Imagem de Impressão Digital"),
    outputs=[gr.Image(label="Preview da Imagem Enviada"), gr.Label(num_top_classes=5)],
    title="Classificador de Impressões Digitais",
    description="Carregue uma imagem de impressão digital para identificar a pessoa.",
)

# Para rodar a interface (se este for o seu arquivo principal)
if __name__ == "__main__":
    interface.launch()
