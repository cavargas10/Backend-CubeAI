from config.firebase_config import db, bucket
from gradio_client import Client
from dotenv import load_dotenv
import datetime
import random
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

client_texto3d_url = os.getenv("CLIENT_TEXTO3D_URL")

client = Client(client_texto3d_url)

def text3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_text3d(user_uid, generation_name, user_prompt, selected_style):
    
    if text3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    full_prompt = f"A {selected_style} 3D render of {user_prompt}, The style should be {selected_style}, with Vibrant colors, emphasizing the essential features and textures. The pose should clearly showcase the full form of the {user_prompt} from a Three-quarter perspective"

    num_steps = random.randint(30, 100)
    seed = random.randint(0, 100000)

    try:
        result_generate_mv = client.predict(
            condition_input_image=None,
            prompt=full_prompt,
            prompt_neg="ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
            input_elevation=0, 
            input_num_steps=num_steps,
            input_seed=seed,
            mv_moedl_option="zero123plus",
            api_name="/generate_mv"
        )

    except Exception as e:
        raise ValueError(f"Error en la generación MV: {e}")

    try:
        result_generate_3d = client.predict(
            condition_input_image=None,
            mv_moedl_option="zero123plus",  
            input_seed=seed,  
            api_name="/generate_3d"
        )
        
    except Exception as e:
        error_message = str(e)
        if "You have exceeded your GPU quota" in error_message:
            raise ValueError("Has excedido el uso de GPU. Por favor, intenta más tarde.")
        elif "None" in error_message:
            raise ValueError("No existe GPU disponibles, inténtalo más tarde")
        else:
            raise ValueError(error_message)

    generation_folder = f'{user_uid}/Texto3D/{generation_name}'

    mv_image_path = result_generate_mv[0] if isinstance(result_generate_mv, tuple) else result_generate_mv
    obj_model_path = result_generate_3d[0] if isinstance(result_generate_3d, tuple) else result_generate_3d

    if not isinstance(mv_image_path, str):
        raise ValueError("Ruta de la imagen MV no es una cadena de texto válida.")
    if not isinstance(obj_model_path, str):
        raise ValueError("Ruta del modelo OBJ no es una cadena de texto válida.")

    try:
        mv_image_url = upload_to_storage(mv_image_path, f'{generation_folder}/mv_image.png')
        obj_url = upload_to_storage(obj_model_path, f'{generation_folder}/model.obj')
    finally:
        if os.path.exists(mv_image_path):
            os.remove(mv_image_path)
        if os.path.exists(obj_model_path):
            os.remove(obj_model_path)
    
    prediction_text3d_result = {
        "generation_name": generation_name,
        "user_prompt": user_prompt,
        "selected_style": selected_style,
        "full_prompt": full_prompt,
        "elevation": 0,  
        "num_steps": num_steps,
        "seed": seed,
        "mv_image": mv_image_url,
        "obj_model": obj_url,
        "timestamp": datetime.datetime.now().isoformat(),
        "prediction_type": "Texto a 3D"
    }

    doc_ref = db.collection('predictions').document(user_uid).collection('Texto3D').document(generation_name)
    doc_ref.set(prediction_text3d_result)

    return prediction_text3d_result

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
