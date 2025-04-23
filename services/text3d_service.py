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
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_text3d(user_uid, generation_name, user_prompt, selected_style):
    if text3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")
    
    full_prompt = f"A {selected_style} 3D render of {user_prompt}. Style: {selected_style}. Emphasize essential features and textures with vibrant colors."

    try:
        client.predict(api_name="/start_session")

        result_get_seed = client.predict(
            randomize_seed=True,
            seed=0,
            api_name="/get_seed"
        )
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
        
        generated_video = result_text_to_3d["video"]
        
        # Verificar que el archivo generado existe
        if not os.path.exists(generated_video):
            raise FileNotFoundError(f"El archivo generado {generated_video} no existe.")
        
        # Paso 4: Extraer el archivo GLB
        result_extract_glb = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb"
        )
        extracted_glb_path = result_extract_glb[1]
        
        # Verificar que el archivo GLB existe
        if not os.path.exists(extracted_glb_path):
            raise FileNotFoundError(f"El archivo GLB {extracted_glb_path} no existe.")
        
        generation_folder = f'{user_uid}/Texto3D/{generation_name}'
        generated_video_url = upload_to_storage(generated_video, f'{generation_folder}/generated_video.mp4')
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')

        prediction_text3d_result = {
            "generation_name": generation_name,
            "selected_style": selected_style,
            "full_prompt": full_prompt,
            "generated_video": generated_video_url,
            "glb_model_t3d": glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Texto a 3D"
        }
        doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
        doc_ref.set(prediction_text3d_result)
        return prediction_text3d_result
    
    except Exception as e:
        error_message = str(e)
        if "You have exceeded your GPU quota" in error_message:
            raise ValueError("Has excedido el uso de GPU. Por favor, intenta más tarde.")
        elif "None" in error_message:
            raise ValueError("No existe GPU disponibles, inténtalo más tarde")
        else:
            raise ValueError(error_message)
            
def get_user_text3d_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Imagen3D')
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