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
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
client_unico3d_url = os.getenv("CLIENT_UNICO3D_URL")
client = create_hf_client(client_unico3d_url)  

def unico3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_unico3d(user_uid, image_file, generation_name):
    if unico3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    unique_filename = f"temp_image_{uuid.uuid4().hex}.png"
    image_file.save(unique_filename)
    
    temp_files_to_clean = [unique_filename]
    extracted_glb_path = None

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
            extracted_glb_path = result_generate3dv2[0]
        else:
            extracted_glb_path = result_generate3dv2
        
        if not extracted_glb_path or not os.path.exists(extracted_glb_path):
             raise FileNotFoundError(f"El archivo GLB no se generó o no se encontró en la ruta: {extracted_glb_path}")
        temp_files_to_clean.append(extracted_glb_path)

        generation_folder = f'{user_uid}/Unico3D/{generation_name}'
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
        
        normalized_result = {
            "generation_name": generation_name,
            "prediction_type": "Unico a 3D",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "modelUrl": glb_url,
            "previewUrl": None, 
            "downloads": [
                {"format": "GLB", "url": glb_url},
            ],
            "raw_data": {} 
        }

        doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
        doc_ref.set(normalized_result)

        return normalized_result
        
    except Exception as e:
        raise

    finally:
        for file_path in temp_files_to_clean:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error al eliminar el archivo temporal {file_path}: {e}")

def add_preview_image(user_uid, generation_name, preview_file):
    doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError(f"No se encontró la generación '{generation_name}' para el usuario.")

    generation_folder = f'{user_uid}/Unico3D/{generation_name}'
    preview_image_url = upload_to_storage(preview_file, f'{generation_folder}/preview_image.png')
    update_data = {"previewImageUrl": preview_image_url}
    doc_ref.update(update_data)
    updated_doc_data = doc.to_dict()
    updated_doc_data.update(update_data) 

    return updated_doc_data

def get_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Unico3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_generation(user_uid, generation_name):
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