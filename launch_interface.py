# launch_interface.py
from interface import interface

if __name__ == "__main__":
    print("🚀 Iniciando interface Gradio para inferência de impressões digitais...")
    interface.launch(share=False)  # share=True se quiser link externo
