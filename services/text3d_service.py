from config.firebase_config import db, bucket
from config.huggingface_config import create_hf_client
from dotenv import load_dotenv
from huggingface_hub import login
import datetime
import os
from utils.storage_utils import upload_to_storage

# Configuración inicial
load_dotenv()

# Autenticación Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

client_texto3d_url = os.getenv("CLIENT_TEXTO3D_URL")
client = create_hf_client(client_texto3d_url)

def text3d_generation_exists(user_uid, generation_name):
    """Verifica si una generación ya existe en Firestore para evitar duplicados."""
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_text3d(user_uid, generation_name, user_prompt, selected_style):
    if text3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    full_prompt = f"A {selected_style} 3D render of {user_prompt}. Style: {selected_style}. Emphasize essential features and textures with vibrant colors."

    temp_files_to_clean = []

    try:
        client.predict(api_name="/start_session")

        result_get_seed = client.predict(randomize_seed=True, seed=0, api_name="/get_seed")
        if not isinstance(result_get_seed, int):
            raise ValueError(f"Seed inválido: {result_get_seed}")
        seed_value = result_get_seed

        result_text_to_3d = client.predict(
            prompt=full_prompt,
            seed=seed_value,
            ss_guidance_strength=7.5,
            ss_sampling_steps=25,
            slat_guidance_strength=7.5,
            slat_sampling_steps=25,
            api_name="/text_to_3d"
        )
        if not isinstance(result_text_to_3d, dict) or "video" not in result_text_to_3d:
            raise ValueError("Error al generar modelo 3D: respuesta de la API inválida.")

        generated_video_path = result_text_to_3d["video"]
        if not os.path.exists(generated_video_path):
            raise FileNotFoundError(f"El archivo de video generado {generated_video_path} no existe.")
        temp_files_to_clean.append(generated_video_path)

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

        generation_folder = f'{user_uid}/Texto3D/{generation_name}'
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
        preview_video_url = upload_to_storage(generated_video_path, f'{generation_folder}/preview.mp4')

        normalized_result = {
            "generation_name": generation_name,
            "prediction_type": "Texto a 3D",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "modelUrl": glb_url,
            "previewUrl": preview_video_url,
            "downloads": [
                {"format": "GLB", "url": glb_url},
            ],
            "raw_data": {
                "user_prompt": user_prompt,
                "selected_style": selected_style,
                "full_prompt_sent_to_api": full_prompt,
            }
        }

        doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
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
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError(f"No se encontró la generación '{generation_name}' para el usuario.")

    generation_folder = f'{user_uid}/Texto3D/{generation_name}'
    preview_image_url = upload_to_storage(preview_file, f'{generation_folder}/preview_image.png')
    update_data = {"previewImageUrl": preview_image_url}
    doc_ref.update(update_data)
    updated_doc_data = doc.to_dict()
    updated_doc_data.update(update_data)

    return updated_doc_data

def get_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Texto3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_generation(user_uid, generation_name):
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