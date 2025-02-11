from config.firebase_config import db, bucket
from config.huggingface_config import create_hf_client
from gradio_client import handle_file
from dotenv import load_dotenv
from huggingface_hub import login
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage

# Configuración inicial
load_dotenv()

# Autenticación Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
client_imagen3d_url = os.getenv("CLIENT_IMAGEN3D_URL")
client = create_hf_client(client_imagen3d_url)  

def generation_exists(user_uid, generation_name):
    """Verifica si una generación ya existe en Firestore"""
    doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
    return doc_ref.get().exists

def create_generation(user_uid, image_file, generation_name):
    if generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")
    
    unique_filename = f"temp_image_{uuid.uuid4().hex}.png"
    image_file.save(unique_filename)
    
    try:
        # Paso 0: Iniciar sesión
        client.predict(api_name="/start_session")
        
        # Paso 1: Preprocesar la imagen
        result_preprocess = client.predict(
            image=handle_file(unique_filename),
            api_name="/preprocess_image"
        )
        preprocess_image_path = result_preprocess
        
        # Verificar que el archivo preprocesado existe
        if not os.path.exists(preprocess_image_path):
            raise FileNotFoundError(f"El archivo preprocesado {preprocess_image_path} no existe.")
        
        # Paso 2: Obtener un seed aleatorio
        result_get_seed = client.predict(
            randomize_seed=True,
            seed=0,
            api_name="/get_seed"
        )
        seed_value = result_get_seed
        
        # Paso 3: Generar el modelo 3D
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
        
        # Verificar que el archivo generado existe
        if not os.path.exists(generated_3d_asset):
            raise FileNotFoundError(f"El archivo generado {generated_3d_asset} no existe.")
        
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
        
        # Subir archivos al almacenamiento
        generation_folder = f'{user_uid}/Imagen3D/{generation_name}'
        preprocess_url = upload_to_storage(preprocess_image_path, f'{generation_folder}/preprocess.png')
        generated_3d_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/generated_3d.mp4')
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
        
        # Guardar los resultados en Firestore
        prediction_img3d_result = {
            "generation_name": generation_name,
            "preprocess": preprocess_url,
            "generated_3d": generated_3d_url,
            "glb_model_i23d": glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Imagen a 3D"
        }
        doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
        doc_ref.set(prediction_img3d_result)
        return prediction_img3d_result
    
    except Exception as e:
        error_message = str(e)
        if "You have exceeded your GPU quota" in error_message:
            raise ValueError("Has excedido el uso de GPU. Por favor, intenta más tarde.")
        elif "None" in error_message:
            raise ValueError("No existe GPU disponibles, inténtalo más tarde")
        else:
            raise ValueError(error_message)
    
    finally:
        # Eliminar archivos temporales solo después de completar todo el proceso
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
            
def get_user_generations(user_uid):
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