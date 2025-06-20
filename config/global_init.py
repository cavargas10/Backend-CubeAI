from huggingface_hub import login
from dotenv import load_dotenv
import os

def initialize_hf_token():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        print("Realizando login en Hugging Face...")
        login(token=HF_TOKEN)
        print("Login en Hugging Face completado.")
    else:
        print("ADVERTENCIA: No se encontró la variable de entorno HF_TOKEN.")