from config.firebase_config import db, bucket
from gradio_client import Client, file
from dotenv import load_dotenv
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

client_imagen3d_url = os.getenv("CLIENT_IMAGEN3D_URL")

client = Client(client_imagen3d_url)

def generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Imagen3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_generation(user_uid, image_file, generation_name):
    if generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    unique_filename = f"temp_image_{uuid.uuid4().hex}.png"
    image_file.save(unique_filename)

    try:
        result_check_input = client.predict(
            input_image=file(unique_filename),
            api_name="/check_input_image"
        )

        result_preprocess = client.predict(
            input_image=file(unique_filename),
            do_remove_background=True,
            api_name="/preprocess"
        )
        preprocess_image_path = result_preprocess

        result_generate_mvs = client.predict(
            input_image=file(preprocess_image_path),
            sample_steps=75,
            sample_seed=42,
            api_name="/generate_mvs"
        )
        generate_mvs_image_path = result_generate_mvs
        
        result_make3d = client.predict(
            api_name="/make3d"
        )

        make3d_obj_path, make3d_glb_path = result_make3d

        generation_folder = f'{user_uid}/{generation_name}'

        preprocess_url = upload_to_storage(preprocess_image_path, f'{generation_folder}/preprocess.png')
        generate_mvs_url = upload_to_storage(generate_mvs_image_path, f'{generation_folder}/generate_mvs.png')
        make3d_obj_url = upload_to_storage(make3d_obj_path, f'{generation_folder}/make3d.obj')
        make3d_glb_url = upload_to_storage(make3d_glb_path, f'{generation_folder}/make3d.glb')

        prediction_img3d_result = {
            "generation_name": generation_name,
            "check_input_image": result_check_input,
            "preprocess": preprocess_url,
            "generate_mvs": generate_mvs_url,
            "make3d": [make3d_obj_url, make3d_glb_url],
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
    
    generation_folder = f"{user_uid}/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True
