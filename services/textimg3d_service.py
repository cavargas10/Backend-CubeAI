from config.firebase_config import db, bucket
from config.huggingface_config import create_hf_client
from gradio_client import handle_file
from dotenv import load_dotenv
from huggingface_hub import login
import datetime
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
client_textimg3d_url = os.getenv("CLIENT_TEXTOIMAGEN3D_URL")
client = create_hf_client(client_textimg3d_url)  

def textimg3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('TextoImagen3D').document(generation_name)
    return doc_ref.get().exists

def create_textimg3d(user_uid, generation_name, subject, style, additional_details):
    if textimg3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")
    
    prompt_generation = f"{subject}, {additional_details}, style {style}, three quarter angle"
    
    temp_files_to_clean = []

    try:
        client.predict(api_name="/start_session")

        generated_image_path = client.predict(
            prompt=prompt_generation,
            seed=42,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=3.5,
            api_name="/generate_flux_image"
        )
        if not generated_image_path or not os.path.exists(generated_image_path):
            raise FileNotFoundError("Error al generar la imagen 2D base.")
        temp_files_to_clean.append(generated_image_path)
        
        preprocess_image_path = client.predict(
            image=handle_file(generated_image_path),
            api_name="/preprocess_image"
        )
        if not preprocess_image_path or not os.path.exists(preprocess_image_path):
            raise FileNotFoundError("Error al preprocesar la imagen generada.")
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
        if not isinstance(result_image_to_3d, dict) or "video" not in result_image_to_3d:
            raise ValueError("Error al generar el modelo 3D: respuesta de la API inválida.")
        generated_3d_asset = result_image_to_3d["video"]
        if not os.path.exists(generated_3d_asset):
            raise FileNotFoundError(f"El archivo 3D generado {generated_3d_asset} no existe.")
        temp_files_to_clean.append(generated_3d_asset)

        result_extract_glb = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb"
        )
        extracted_glb_path = result_extract_glb[1]
        if not os.path.exists(extracted_glb_path):
            raise FileNotFoundError(f"El archivo GLB extraído {extracted_glb_path} no existe.")
        temp_files_to_clean.append(extracted_glb_path)

        client.predict(api_name="/end_session")
        
        generation_folder = f'{user_uid}/TextoImagen3D/{generation_name}'
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
        preview_video_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/preview.mp4')
        generated_2d_image_url = upload_to_storage(generated_image_path, f'{generation_folder}/generated_2d_image.png')

        normalized_result = {
            "generation_name": generation_name,
            "prediction_type": "Texto a Imagen a 3D",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "modelUrl": glb_url,
            "previewUrl": preview_video_url,
            "downloads": [
                {"format": "GLB", "url": glb_url},
            ],
            "raw_data": {
                "prompt": prompt_generation,
                "generated_2d_image_url": generated_2d_image_url
            }
        }
        
        doc_ref = db.collection('predictions').document(user_uid).collection('TextoImagen3D').document(generation_name)
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

def get_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('TextoImagen3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('TextoImagen3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False

    generation_folder = f"{user_uid}/TextoImagen3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()
    doc_ref.delete()
    return True