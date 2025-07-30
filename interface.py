# interface.py
import gradio as gr
from utils import preprocess_image, load_model
from dataset import FingerprintDataset
from config import *
import torch # Adicionado para o exemplo funcionar

# Carrega classes e modelo treinado
dummy_dataset = FingerprintDataset(TRAIN_DIR)
class_names = dummy_dataset.classes
model = load_model(len(class_names))

def classify(image):
    tensor = preprocess_image(image)
    probabilities = model(tensor.to(DEVICE))
    probs = torch.nn.functional.softmax(probabilities, dim=1).cpu().detach().numpy().flatten()
    confidences = {cls: float(prob) for cls, prob in zip(class_names, probs)}
    
    # ALTERAÇÃO 1: Retornar a imagem original junto com as classificações.
    # A função agora retorna uma tupla: (imagem, dicionário_de_confianças)
    return image, confidences

# ALTERAÇÃO 2: Atualizar os outputs para receber uma imagem e um label.
interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Imagem de Impressão Digital"),
    outputs=[
        gr.Image(label="Preview da Imagem Enviada"), 
        gr.Label(num_top_classes=5)
    ],
    title="Classificador de Impressões Digitais",
    description="Carregue uma imagem de impressão digital para identificar a pessoa."
)

# Para rodar a interface (se este for o seu arquivo principal)
if __name__ == "__main__":
    interface.launch()