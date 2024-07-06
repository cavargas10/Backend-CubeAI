from config.firebase_config import db, bucket
from gradio_client import Client, file
from dotenv import load_dotenv
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

client = Client("cavargas10/Unico3D")

def unico3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
    doc = doc_ref.get()
    return doc.exists

def create_unico3d_generation(user_uid, image_file, generation_name):
    if unico3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

    unique_filename = f"temp_image_{uuid.uuid4().hex}.png"
    image_file.save(unique_filename)

    try:
        result = client.predict(
            preview_img=file(unique_filename),
            input_processing=True,
            seed=-1,
            render_video=False,
            do_refine=True,
            expansion_weight=0.1,
            init_type="std",
            api_name="/generate3dv2"
        )

        # Imprimir el resultado para depuración
        print(f"Resultado de la predicción: {result}")

        # Asumiendo que el resultado es una tupla con la ruta del archivo .glb como primer elemento
        if isinstance(result, tuple) and len(result) > 0:
            glb_path = result[0]
        else:
            raise ValueError(f"Resultado inesperado de la predicción: {result}")

        # Verificar si glb_path es una ruta válida
        if not isinstance(glb_path, (str, bytes, os.PathLike)):
            raise ValueError(f"La ruta del archivo GLB no es válida: {glb_path}")

        generation_folder = f'{user_uid}/{generation_name}'
        glb_url = upload_to_storage(glb_path, f'{generation_folder}/model.glb')

        prediction_result = {
            "generation_name": generation_name,
            "glb_model": glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Unico3D"
        }

        doc_ref = db.collection('predictions').document(user_uid).collection('Unico3D').document(generation_name)
        doc_ref.set(prediction_result)

        return prediction_result

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

    generation_folder = f"{user_uid}/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True