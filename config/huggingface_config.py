from dotenv import load_dotenv
import os
import gradio_client

load_dotenv()

def create_hf_client(url):
    try:
        return gradio_client.Client(
            url,
            headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}  # <-- Añade esto
        )
    except Exception as e:
        print(f"Error al crear cliente HF para {url}: {e}")
        raise