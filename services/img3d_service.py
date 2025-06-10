from config.firebase_config import db, bucket
from config.huggingface_config import create_hf_client
from gradio_client import handle_file
from dotenv import load_dotenv
from huggingface_hub import login
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

client_imagen3d_url = os.getenv("CLIENT_IMAGEN3D_URL")
client = create_hf_client(client_imagen3d_url)  

def img3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_generation(user_uid, image_file, generation_name):
    if img3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    unique_filename = f"temp_image_{uuid.uuid4().hex}.png"
    image_file.save(unique_filename)
    
    temp_files_to_clean = [unique_filename]

    try:
        client.predict(api_name="/start_session")
        
        result_preprocess = client.predict(
            image=handle_file(unique_filename),
            api_name="/preprocess_image"
        )
        preprocess_image_path = result_preprocess
        if not os.path.exists(preprocess_image_path):
            raise FileNotFoundError(f"El archivo preprocesado {preprocess_image_path} no existe.")
        temp_files_to_clean.append(preprocess_image_path)
        
        result_get_seed = client.predict(randomize_seed=True, seed=0, api_name="/get_seed")
        seed_value = result_get_seed
        
        result_image_to_3d = client.predict(
            image=handle_file(preprocess_image_path),
            seed=seed_value,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3,
            slat_sampling_steps=12,
            api_name="/image_to_3d"
        )
        generated_3d_asset = result_image_to_3d["video"]
        if not os.path.exists(generated_3d_asset):
            raise FileNotFoundError(f"El archivo generado {generated_3d_asset} no existe.")
        temp_files_to_clean.append(generated_3d_asset)

        result_extract_glb = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb"
        )
        extracted_glb_path = result_extract_glb[1]
        if not os.path.exists(extracted_glb_path):
            raise FileNotFoundError(f"El archivo GLB {extracted_glb_path} no existe.")
        temp_files_to_clean.append(extracted_glb_path)
        
        client.predict(api_name="/end_session")

        generation_folder = f'{user_uid}/Imagen3D/{generation_name}'
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')        
        preview_video_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/preview.mp4')       
        preprocess_url = upload_to_storage(preprocess_image_path, f'{generation_folder}/preprocess.png')
        normalized_result = {
            "generation_name": generation_name,
            "prediction_type": "Imagen a 3D",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "modelUrl": glb_url,
            "previewUrl": preview_video_url,
            
            "downloads": [
                {"format": "GLB", "url": glb_url},
            ],
    
            "raw_data": {
                "preprocess_image_url": preprocess_url
            }
        }
        
        doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
        doc_ref.set(normalized_result)
        
        return normalized_result
    
    except Exception as e:
        error_message = str(e)
        if "You have exceeded your GPU quota" in error_message:
            raise ValueError("Has excedido tu cuota de uso de GPU. Por favor, intenta más tarde.")
        elif "None" in error_message: 
            raise ValueError("No hay GPUs disponibles en este momento, por favor inténtalo más tarde.")
        else:
            raise
    
    finally:
        for file_path in temp_files_to_clean:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error al eliminar el archivo temporal {file_path}: {e}")

def add_preview_image(user_uid, generation_name, preview_file):
    doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError(f"No se encontró la generación '{generation_name}' para el usuario.")

    generation_folder = f'{user_uid}/Imagen3D/{generation_name}'
    preview_image_url = upload_to_storage(preview_file, f'{generation_folder}/preview_image.png')
    update_data = {"previewImageUrl": preview_image_url}
    doc_ref.update(update_data)
    updated_doc_data = doc.to_dict()
    updated_doc_data.update(update_data) 

    return updated_doc_data

def get_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Imagen3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    
    generation_folder = f"{user_uid}/Imagen3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True