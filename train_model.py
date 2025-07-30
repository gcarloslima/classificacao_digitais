"""
Script principal para iniciar o treinamento do modelo.

Este módulo é o ponto de entrada para executar o treinamento do modelo
de classificação de impressões digitais. Ele importa e executa a função
de treinamento principal, exibindo mensagens informativas sobre o progresso.
"""

from train import train_model

if __name__ == "__main__":
    print("🔧 Iniciando treinamento do modelo de impressão digital...\n")
    model, classes = train_model()
    print("\n✅ Treinamento concluído com sucesso!")
