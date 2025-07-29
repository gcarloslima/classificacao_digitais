# train_model.py
from train import train_model

if __name__ == "__main__":
    print("🔧 Iniciando treinamento do modelo de impressão digital...\n")
    model, classes = train_model()
    print("\n✅ Treinamento concluído com sucesso!")
