from config.firebase_config import db, bucket
from config.huggingface_config import create_hf_client
from dotenv import load_dotenv
from huggingface_hub import login  # <-- Añade esto
import datetime
import random
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

# Autentica con Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)  # <-- Esto valida tu token

client_texto3d_url = os.getenv("CLIENT_TEXTO3D_URL")
client = create_hf_client(client_texto3d_url)

def text3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_text3d(user_uid, generation_name, user_prompt, selected_style):
    # Check if generation name already exists
    if text3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    # Prepare the full prompt
    full_prompt = f"A {selected_style} 3D render of {user_prompt}, The style should be {selected_style}, with Vibrant colors, emphasizing the essential features and textures."

    # Generate random steps and seed
    num_steps = random.randint(30, 100)
    seed = random.randint(0, 100000)

    # Initialize the Gradio client using the create_hf_client function
    client = create_hf_client(client_texto3d_url)

    try:
        # Paso 1: Generación 3D con "/generation_all"
        result_generate = client.predict(
            caption=full_prompt,
            steps=num_steps,
            guidance_scale=5.5,
            seed=seed,
            octree_resolution="256",
            api_name="/generation_all"
        )
        
        from gradio_client import Client, file

        # Extrae el GLB de la TUPLA (índice 1)
        glb_model_path = result_generate[1]  # <-- Aquí está el cambio clave

    except Exception as e:
        error_message = str(e)
        if "exceeded your GPU quota" in error_message:
            raise ValueError("Límite de GPU excedido. Actualiza tu cuenta Hugging Face.")
        else:
            raise ValueError(f"Error en generación 3D: {error_message}")

    except Exception as e:
        raise ValueError(f"Error en la generación 3D: {e}")

    # Prepare generation folder
    generation_folder = f'{user_uid}/Texto3D/{generation_name}'

    try:
        # Upload GLB model to storage
        glb_url = upload_to_storage(glb_model_path, f'{generation_folder}/model.glb')
    finally:
        # Clean up local file
        if os.path.exists(glb_model_path):
            os.remove(glb_model_path)
    
    # Prepare prediction result
    prediction_text3d_result = {
        "generation_name": generation_name,
        "user_prompt": user_prompt,
        "selected_style": selected_style,
        "full_prompt": full_prompt,
        "num_steps": num_steps,
        "seed": seed,
        "glb_model": glb_url,
        "timestamp": datetime.datetime.now().isoformat(),
        "prediction_type": "Texto a 3D"
    }

    # Save to Firestore
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc_ref.set(prediction_text3d_result)

    return prediction_text3d_result

# Resto de las funciones permanecen igual
def get_user_text3d_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Texto3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_text3d_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    
    generation_folder = f"{user_uid}/Texto3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True