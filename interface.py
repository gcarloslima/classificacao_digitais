# interface.py
import gradio as gr
from utils import preprocess_image, load_model
from dataset import FingerprintDataset
from config import *

# Carrega classes e modelo treinado
dummy_dataset = FingerprintDataset(TRAIN_DIR)
class_names = dummy_dataset.classes
model = load_model(len(class_names))

def classify(image):
    tensor = preprocess_image(image)
    probabilities = model(tensor.to(DEVICE))
    probs = torch.nn.functional.softmax(probabilities, dim=1).cpu().detach().numpy().flatten()
    topk = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)[:5]
    return {cls: float(prob) for cls, prob in topk}

interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Imagem de Impressão Digital"),
    outputs=gr.Label(num_top_classes=5),
    title="Classificador de Impressões Digitais",
    description="Carregue uma imagem de impressão digital para identificar a pessoa."
)
