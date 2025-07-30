"""
Script de lançamento da interface Gradio.

Este módulo é responsável por inicializar e executar a interface web
para classificação de impressões digitais. Ele importa a interface
já configurada e a executa com opções de compartilhamento.
"""

from interface import interface

if __name__ == "__main__":
    print("🚀 Iniciando interface Gradio para inferência de impressões digitais...")
    interface.launch(share=True)  # share=True se quiser link externo
