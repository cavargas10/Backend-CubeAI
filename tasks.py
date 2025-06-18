# Backend-CubeAI/tasks.py

import os
from celery_app import celery
from services.text3d_service import text3d_service # Importamos el servicio existente

# Es importante inicializar las dependencias que las tareas necesitan,
# como el login de Hugging Face, DENTRO del worker.
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)


# El decorador @celery.task convierte esta función normal en una tarea de Celery.
@celery.task
def run_text3d_generation(user_uid, generation_name, user_prompt, selected_style):
    """
    Tarea asíncrona que ejecuta la generación de Texto a 3D.
    Esta función se ejecutará en segundo plano en un proceso 'worker'.
    """
    try:
        # Llamamos a la función original del servicio que ya tienes.
        # ¡No hemos tenido que reescribir la lógica de generación!
        print(f"Iniciando generación de texto a 3D para {user_uid} con nombre '{generation_name}'")
        
        result = text3d_service.create_text3d(user_uid, generation_name, user_prompt, selected_style)
        
        print(f"Generación '{generation_name}' completada con éxito.")
        return result
        
    except Exception as e:
        # Si algo sale mal, Celery guardará la excepción.
        # Es buena idea imprimir el error también en el log del worker.
        print(f"ERROR en la generación '{generation_name}': {e}")
        # Al hacer 'raise', Celery marcará la tarea como 'FAILURE'.
        raise e