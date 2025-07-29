from train import train_model
from interface import interface

if __name__ == "__main__":
    # Treina e salva o modelo
    train_model()

    # Depois de treinar, inicia a interface Gradio
    interface.launch()
