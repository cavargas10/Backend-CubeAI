from config.firebase_config import db, bucket
from gradio_client import Client, file
from config.huggingface_config import create_hf_client
from huggingface_hub import login
from dotenv import load_dotenv
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage

load_dotenv()
# Autenticación Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
client_unico3d_url = os.getenv("CLIENT_UNICO3D_URL")
client = create_hf_client(client_unico3d_url)  # Serialización activada

def unico3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_unico3d(user_uid, image_file, generation_name):
    if unico3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    unique_filename = f"temp_image_{uuid.uuid4().hex}.png"
    image_file.save(unique_filename)

    try:
        result_generate3dv2 = client.predict(
            file(unique_filename),
            True,
            -1,
            False,
            True,
            0.1,
            "std",
            "/generate3dv2"
        )

        if isinstance(result_generate3dv2, tuple):
            obj_glb_path = result_generate3dv2[0]  
        else:
            obj_glb_path = result_generate3dv2

        generation_folder = f'{user_uid}/Unico3D/{generation_name}'

        obj_glb_url = upload_to_storage(obj_glb_path, f'{generation_folder}/obj_glb.glb')

        prediction_unico3d_result = {
            "generation_name": generation_name,
            "obj_glb": obj_glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Unico a 3D"
        }

        doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
        doc_ref.set(prediction_unico3d_result)

        return prediction_unico3d_result
    except Exception as e:
        error_message = str(e)
        if "You have exceeded your GPU quota" in error_message:
            raise ValueError("Has excedido el uso de GPU. Por favor, intenta más tarde.")
        elif "None" in error_message:
            raise ValueError("No existe GPU disponibles, inténtalo más tarde")
        else:
            raise ValueError(error_message)
    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)

def get_user_unico3d_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Unico3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_unico3d_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    
    generation_folder = f"{user_uid}/Unico3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True