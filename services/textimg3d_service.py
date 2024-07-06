from config.firebase_config import db, bucket
from gradio_client import Client, file
from dotenv import load_dotenv
import datetime
import random
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

client_textimg3d_url = os.getenv("CLIENT_TEXTOIMAGEN3D_URL")

client = Client(client_textimg3d_url)

def textimg3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('TextImg3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_textimg3d(user_uid, generation_name, subject, style, additional_details):
    
    if textimg3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    try:
        result_generate_image = client.predict(
            subject=subject,
            style=style,
            color_scheme="Vibrant",
            angle="Three-quarter",
            lighting_type="Bright and Even",
            additional_details=additional_details,
            api_name="/generate_image"
        )

    except Exception as e:
        raise ValueError(f"Error en la generación de imagen: {e}")

    image_path = result_generate_image[0] if isinstance(result_generate_image, tuple) else result_generate_image

    if not isinstance(image_path, str):
        raise ValueError("Ruta de la imagen no es una cadena de texto válida.")

    try:
        processed_image = client.predict(
            input_image=file(image_path),
            api_name="/preprocess"
        )

        checked_image = client.predict(
            processed_image=file(processed_image),
            api_name="/check_image"
        )

        mv_result = client.predict(
            input_image=file(processed_image),
            api_name="/generate_mvs"
        )
        
        mv_result_path = mv_result
        final_result = client.predict(
            api_name="/make3d"
        )

    except Exception as e:
        raise ValueError(f"Error en la generación 3D: {e}")
    make3d_obj_path, make3d_glb_path = final_result
    generation_folder = f'{user_uid}/TextImg3D/{generation_name}'
    
    try:
        image_path = upload_to_storage(result_generate_image, f'{generation_folder}/mv_image.png')
        mv_image_url = upload_to_storage(mv_result_path, f'{generation_folder}/mv_image.png')
        generate_image_url = upload_to_storage(result_generate_image, f'{generation_folder}/generate_image.png')
        final_obj_url  = upload_to_storage(make3d_obj_path, f'{generation_folder}/make3d.obj')
        final_glb_url  = upload_to_storage(make3d_glb_path, f'{generation_folder}/make3d.glb')
    finally:
        if os.path.exists(mv_result_path):
            os.remove(mv_result_path)
        if os.path.exists(make3d_obj_path):
            os.remove(make3d_obj_path)

    prediction_result = {
        "generation_name": generation_name,
        "subject": subject,
        "style": style,
        "additional_details": additional_details,
        "mv_image": mv_image_url,
        "generate_image": generate_image_url,
        "make3d": [final_obj_url, final_glb_url],
        "timestamp": datetime.datetime.now().isoformat(),
        "prediction_type": "Texto a Imagen a 3D"
    }

    doc_ref = db.collection('predictions').document(user_uid).collection('TextImg3D').document(generation_name)
    doc_ref.set(prediction_result)

    return prediction_result

def get_user_textimg3d_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('TextImg3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_textimg3d_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('TextImg3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    
    generation_folder = f"{user_uid}/TextImg3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True
