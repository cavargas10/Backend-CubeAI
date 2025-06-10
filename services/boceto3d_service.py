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
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
client_boceto3d_url = os.getenv("CLIENT_BOCETO3D_URL")
client = create_hf_client(client_boceto3d_url)

def boceto3d_generation_exists(user_uid, generation_name):
    """Verifica si una generación ya existe en Firestore"""
    doc_ref = db.collection('predictions').document(user_uid).collection('Boceto3D').document(generation_name)
    return doc_ref.get().exists

def create_boceto3d(user_uid, image_file, generation_name, description=""):
    
    print("Iniciando sesión con la API...")
    client.predict(api_name="/start_session")
        
    if boceto3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")
    
    temp_files = []
    try:
        unique_filename = f"temp_boceto_{uuid.uuid4().hex}.png"
        image_file.save(unique_filename)
        temp_files.append(unique_filename)
        
        print(f"Preprocesando la imagen...")
        preprocess_result = client.predict(
            image={"background": handle_file(unique_filename),
                   "layers": [handle_file(unique_filename)],
                   "composite": handle_file(unique_filename)},
            prompt=description or "Objeto 3D",
            negative_prompt="",
            style_name="3D Model",
            num_steps=8,
            guidance_scale=5,
            controlnet_conditioning_scale=0.85,
            api_name="/preprocess_image"
        )

        if not preprocess_result:
            raise ValueError("Error en el preprocesamiento: respuesta vacía.")

        processed_image_path = preprocess_result if isinstance(preprocess_result, str) else preprocess_result.get("image")
        if not processed_image_path:
            raise ValueError("El preprocesamiento no devolvió imágenes válidas.")

        temp_files.append(processed_image_path)
        print(f"Imagen preprocesada")

        print(f"Obteniendo seed aleatorio...")
        result_get_seed = client.predict(randomize_seed=True, seed=0, api_name="/get_seed")
        seed_value = result_get_seed.get("seed", 0) if isinstance(result_get_seed, dict) else result_get_seed

        print(f"Generando modelo 3D...")
        result_image_to_3d = client.predict(
            image=handle_file(processed_image_path),  
            seed=seed_value,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3,
            slat_sampling_steps=12,
            api_name="/image_to_3d"
        )

        if not isinstance(result_image_to_3d, dict) or "video" not in result_image_to_3d:
            raise ValueError("Error en la generación 3D: respuesta inválida.")

        generated_3d_asset = result_image_to_3d["video"]
        if not generated_3d_asset:
            raise ValueError("El modelo 3D no se generó correctamente.")
        
        temp_files.append(generated_3d_asset)
        print(f"Modelo 3D generado")

        print(f"Extrayendo GLB...")
        result_extract_glb = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb"
        )
        extracted_glb_path = result_extract_glb[1]
        
        # Verificar que el archivo GLB existe
        if not os.path.exists(extracted_glb_path):
            raise FileNotFoundError(f"El archivo GLB {extracted_glb_path} no existe.")
        
        print(f"Archivo GLB extraído en: {extracted_glb_path}")
        if not extracted_glb_path or not isinstance(extracted_glb_path, str):
            raise ValueError("Error al extraer GLB: respuesta inválida.")
        
        # Paso 5: Finalizar procesamiento
        print("Finalizar sesión con la API...")
        client.predict(api_name="/end_session")

        print(f"Subiendo archivos al almacenamiento...")
        generation_folder = f'{user_uid}/Boceto3D/{generation_name}'
        processed_url = upload_to_storage(processed_image_path, f'{generation_folder}/processed_image.png')
        generated_3d_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/generated_3d.mp4')
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')

        print(f"Guardando resultados en Firestore...")
        prediction_boceto3d_result = {
            "generation_name": generation_name,
            "description": description,
            "processed_image": processed_url,
            "generated_3d": generated_3d_url,
            "glb_model_b3d": glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Boceto a 3D"
        }
        doc_ref = db.collection('predictions').document(user_uid).collection('Boceto3D').document(generation_name)
        doc_ref.set(prediction_boceto3d_result)

        print(f"Proceso completado con éxito.")
        return prediction_boceto3d_result

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error desconocido: {str(e)}")
        raise
    finally:
        for file in temp_files:
            if file and isinstance(file, str) and os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"No se pudo eliminar {file}: {str(e)}")
        print(f"Archivos temporales eliminados.")

def get_generations(user_uid):
    generations_ref = db.collection('predictions').document(user_uid).collection('Boceto3D')
    return [gen.to_dict() for gen in generations_ref.stream()]

def delete_generation(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('Boceto3D').document(generation_name)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    
    generation_folder = f"{user_uid}/Boceto3D/{generation_name}"
    blobs = bucket.list_blobs(prefix=generation_folder)
    for blob in blobs:
        blob.delete()
    doc_ref.delete()
    return True