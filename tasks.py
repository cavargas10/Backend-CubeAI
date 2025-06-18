# Backend-CubeAI/tasks.py

import os
from celery_app import celery
from services.text3d_service import text3d_service # Importamos el servicio existente
# New service imports
from services.img3d_service import img3d_service
from services.textimg3d_service import textimg3d_service
from services.unico3d_service import unico3d_service
from services.multiimg3d_service import multiimg3d_service
from services.boceto3d_service import boceto3d_service

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

# New tasks below

@celery.task
def run_img3d_generation(user_uid, image_file_path, generation_name):
    """
    Tarea asíncrona que ejecuta la generación de Imagen a 3D.
    """
    try:
        print(f"Iniciando generación de Imagen a 3D para {user_uid} con nombre '{generation_name}' usando el archivo {image_file_path}")
        # Assuming img3d_service.create_generation can handle image_file_path
        # This might require modification in img3d_service to open and read the file from the path,
        # or to receive bytes directly if that's how it's adapted.
        # For this implementation, I will pass the path directly as per subtask guidance.
        result = img3d_service.create_generation(user_uid, image_file_path, generation_name)
        print(f"Generación Imagen a 3D '{generation_name}' completada con éxito.")
        return result
    except Exception as e:
        print(f"ERROR en la generación Imagen a 3D '{generation_name}': {e}")
        raise e

@celery.task
def run_textimg3d_generation(user_uid, generation_name, subject, style, additional_details):
    """
    Tarea asíncrona que ejecuta la generación de Texto+Imagen a 3D.
    """
    try:
        print(f"Iniciando generación de Texto+Imagen a 3D para {user_uid} con nombre '{generation_name}'")
        result = textimg3d_service.create_textimg3d(user_uid, generation_name, subject, style, additional_details)
        print(f"Generación Texto+Imagen a 3D '{generation_name}' completada con éxito.")
        return result
    except Exception as e:
        print(f"ERROR en la generación Texto+Imagen a 3D '{generation_name}': {e}")
        raise e

@celery.task
def run_unico3d_generation(user_uid, image_file_path, generation_name):
    """
    Tarea asíncrona que ejecuta la generación de Unico3D (Imagen única a 3D).
    """
    try:
        print(f"Iniciando generación de Unico3D para {user_uid} con nombre '{generation_name}' usando el archivo {image_file_path}")
        # Similar assumption as run_img3d_generation for image_file_path
        result = unico3d_service.create_unico3d(user_uid, image_file_path, generation_name)
        print(f"Generación Unico3D '{generation_name}' completada con éxito.")
        return result
    except Exception as e:
        print(f"ERROR en la generación Unico3D '{generation_name}': {e}")
        raise e

@celery.task
def run_multiimg3d_generation(user_uid, frontal_image_path, lateral_image_path, trasera_image_path, generation_name):
    """
    Tarea asíncrona que ejecuta la generación de Multi-Imagen a 3D.
    """
    try:
        print(f"Iniciando generación de Multi-Imagen a 3D para {user_uid} con nombre '{generation_name}'")
        # Similar assumption as run_img3d_generation for image paths
        result = multiimg3d_service.create_multiimg3d(user_uid, frontal_image_path, lateral_image_path, trasera_image_path, generation_name)
        print(f"Generación Multi-Imagen a 3D '{generation_name}' completada con éxito.")
        return result
    except Exception as e:
        print(f"ERROR en la generación Multi-Imagen a 3D '{generation_name}': {e}")
        raise e

@celery.task
def run_boceto3d_generation(user_uid, image_file_path, generation_name, description):
    """
    Tarea asíncrona que ejecuta la generación de Boceto a 3D.
    """
    try:
        print(f"Iniciando generación de Boceto a 3D para {user_uid} con nombre '{generation_name}' usando el archivo {image_file_path}")
        # Similar assumption as run_img3d_generation for image_file_path
        result = boceto3d_service.create_boceto3d(user_uid, image_file_path, generation_name, description)
        print(f"Generación Boceto a 3D '{generation_name}' completada con éxito.")
        return result
    except Exception as e:
        print(f"ERROR en la generación Boceto a 3D '{generation_name}': {e}")
        raise e