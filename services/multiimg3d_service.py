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
client_multiimg3d_url = os.getenv("CLIENT_MULTI3D_URL")
client = create_hf_client(client_multiimg3d_url)  

def multiimg3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('MultiImagen3D').document(generation_name)
    return doc_ref.get().exists

def create_multiimg3d(user_uid, frontal_image, lateral_image, trasera_image, generation_name):
    if multiimg3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")
    
    # Guardar las imágenes temporales
    print(f"Guardando las imágenes temporales...")
    unique_filenames = {
        "frontal": f"temp_frontal_{uuid.uuid4().hex}.png",
        "lateral": f"temp_lateral_{uuid.uuid4().hex}.png",
        "trasera": f"temp_trasera_{uuid.uuid4().hex}.png"
    }
    frontal_image.save(unique_filenames["frontal"])
    lateral_image.save(unique_filenames["lateral"])
    trasera_image.save(unique_filenames["trasera"])
    
    try:
        # Paso 0: Iniciar sesión
        print("Iniciando sesión con la API...")
        client.predict(api_name="/start_session")
        
        # Paso 1: Preprocesar las imágenes (si es necesario)
        print("Preprocesando las imágenes...")
        preprocess_results = client.predict(
            images=[
                {"image": handle_file(unique_filenames["frontal"])},
                {"image": handle_file(unique_filenames["lateral"])},
                {"image": handle_file(unique_filenames["trasera"])}
            ],
            api_name="/preprocess_images"
        )
        
        # Extraer las rutas de las imágenes preprocesadas
        preprocess_paths = [
            preprocess_results[0]["image"],
            preprocess_results[1]["image"],
            preprocess_results[2]["image"]
        ]
        
        # Validar que el resultado del preprocesamiento sea una lista válida
        if not isinstance(preprocess_paths, list) or len(preprocess_paths) != 3:
            raise ValueError("Error al preprocesar las imágenes: respuesta inválida.")
        
        # Paso 2: Obtener un seed aleatorio
        print("Obteniendo seed aleatorio...")
        result_get_seed = client.predict(
            randomize_seed=True,
            seed=0,
            api_name="/get_seed"
        )
        seed_value = result_get_seed
        
        # Validar que el seed sea un valor válido
        if not isinstance(seed_value, int):
            raise ValueError(f"Seed inválido: {seed_value}")
        print(f"Seed generado: {seed_value}")
        
        # Paso 3: Generar el modelo 3D usando las tres imágenes preprocesadas
        print("Generando modelo 3D...")
        result_image_to_3d = client.predict(
            multiimages=[
                {"image": handle_file(preprocess_paths[0])},
                {"image": handle_file(preprocess_paths[1])},
                {"image": handle_file(preprocess_paths[2])}
            ],
            seed=seed_value,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3,
            slat_sampling_steps=12,
            multiimage_algo="stochastic",
            api_name="/image_to_3d"
        )
        generated_3d_asset = result_image_to_3d["video"]  # Acceder al video generado
        
        # Validar que el resultado del modelo 3D sea válido
        if not result_image_to_3d or len(result_image_to_3d) < 2:
            raise ValueError("Error al generar el modelo 3D: respuesta inválida.")
        
        # Validar que el archivo generado exista
        if not isinstance(generated_3d_asset, str) or not os.path.exists(generated_3d_asset):
            raise FileNotFoundError(f"El archivo generado {generated_3d_asset} no existe.")
        print(f"Modelo 3D generado en: {generated_3d_asset}")
        
        # Paso 4: Extraer el archivo GLB
        result_extract_glb = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb"
        )
        extracted_glb_path = result_extract_glb[1]
        
        if not extracted_glb_path or not isinstance(extracted_glb_path, str):
            raise ValueError("Error al extraer GLB: respuesta inválida.")
        
        # Verificar que el archivo GLB existe
        if not os.path.exists(extracted_glb_path):
            raise FileNotFoundError(f"El archivo GLB {extracted_glb_path} no existe.")
        print(f"Archivo GLB extraído en: {extracted_glb_path}")
        
        # Paso 5: Finalizar procesamiento
        print("Finalizar sesión con la API...")
        client.predict(api_name="/end_session")
        
        # Subir archivos al almacenamiento
        print("Subiendo archivos al almacenamiento...")
        generation_folder = f'{user_uid}/MultiImagen3D/{generation_name}'
        preprocess_urls = {
            "frontal": upload_to_storage(preprocess_paths[0], f'{generation_folder}/preprocess_frontal.png'),
            "lateral": upload_to_storage(preprocess_paths[1], f'{generation_folder}/preprocess_lateral.png'),
            "trasera": upload_to_storage(preprocess_paths[2], f'{generation_folder}/preprocess_trasera.png')
        }
        generated_3d_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/generated_3d.mp4')
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
        
        # Guardar los resultados en Firestore
        print("Guardando resultados en Firestore...")
        prediction_multiimg3d_result = {
            "generation_name": generation_name,
            "preprocess": preprocess_urls,
            "generated_3d": generated_3d_url,
            "glb_model_multi3d": glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Imagen a 3D"
        }
        doc_ref = db.collection('predictions').document(user_uid).collection('MultiImagen3D').document(generation_name)
        doc_ref.set(prediction_multiimg3d_result)
        print("Proceso completado con éxito.")
        return prediction_multiimg3d_result
    
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
        for filename in unique_filenames.values():
            if os.path.exists(filename):
                os.remove(filename)
            
def get_user_multiimg3d_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('MultiImagen3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_multiimg3d_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('MultiImagen3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    
    generation_folder = f"{user_uid}/MultiImagen3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()

    doc_ref.delete()
    return True