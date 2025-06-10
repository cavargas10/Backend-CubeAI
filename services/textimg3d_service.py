from config.firebase_config import db, bucket
from config.huggingface_config import create_hf_client
from gradio_client import handle_file
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
client_textimg3d_url = os.getenv("CLIENT_TEXTOIMAGEN3D_URL")
client = create_hf_client(client_textimg3d_url)  

def textimg3d_generation_exists(user_uid, generation_name):
    doc_ref = db.collection('predictions').document(user_uid).collection('TextoImagen3D').document(generation_name)
    return doc_ref.get().exists

def create_textimg3d(user_uid, generation_name, subject, style, additional_details):
    if textimg3d_generation_exists(user_uid, generation_name):
        raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")
    
    prompt_generation = f"{subject}, {additional_details}, style {style}, three quarter angle"

    try:
        # Paso 0: Iniciar sesión
        print("Iniciando sesión con la API...")
        client.predict(api_name="/start_session")

        # Paso 1: Generar la imagen a partir del texto
        print(f"Generando imagen con prompt: {prompt_generation}")
        result_generate_image = client.predict(
            prompt=prompt_generation,
            seed=42,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=3.5,
            api_name="/generate_flux_image"
        )
        
        # Validación: La respuesta no debe ser None ni de tipo incorrecto
        if not result_generate_image or not isinstance(result_generate_image, str):
            raise ValueError("Error al generar la imagen: respuesta inválida.")
        
        generated_image_path = result_generate_image
        print(f"Imagen generada en: {generated_image_path}")

        # Verificar que el archivo generado existe
        if not os.path.exists(generated_image_path):
            raise FileNotFoundError(f"El archivo generado {generated_image_path} no existe.")

        # Paso 2: Obtener un seed aleatorio
        print("Obteniendo seed aleatorio...")
        result_get_seed = client.predict(
            randomize_seed=True,
            seed=0,
            api_name="/get_seed"
        )

        # Validar que el seed es un número entero
        if not isinstance(result_get_seed, int):
            raise ValueError(f"Seed inválido: {result_get_seed}")

        seed_value = result_get_seed
        print(f"Seed obtenido: {seed_value}")

        # Paso 3: Preprocesar la imagen
        print("Preprocesando imagen...")

        result_preprocess = client.predict(
            image=handle_file(generated_image_path),
            api_name="/preprocess_image"
        )
        preprocess_image_path = result_preprocess
        
        # Validar que el resultado del preprocesamiento sea una cadena válida
        if not isinstance(preprocess_image_path, str) or not preprocess_image_path:
            raise ValueError("Error al preprocesar la imagen.")
        
        if not result_preprocess or not isinstance(result_preprocess, str):
            raise ValueError("Error al preprocesar la imagen: respuesta inválida.")

        # Verificar que el archivo preprocesado existe
        if not os.path.exists(preprocess_image_path):
            raise FileNotFoundError(f"El archivo preprocesado {preprocess_image_path} no existe.")

        print(f"Imagen preprocesada en: {preprocess_image_path}")
        
        # Paso 4: Generar el modelo 3D
        print("Generando modelo 3D...")
        result_image_to_3d = client.predict(
            image=handle_file(preprocess_image_path),
            seed=1472272503,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3,
            slat_sampling_steps=12,
            api_name="/image_to_3d"
        )

        if not result_image_to_3d or not isinstance(result_image_to_3d, dict):
            raise ValueError("Error al generar modelo 3D: respuesta inválida.")

        generated_3d_asset = result_image_to_3d.get("video")

        if not isinstance(generated_3d_asset, str) or not generated_3d_asset:
            raise ValueError("Error: la generación del modelo 3D no devolvió un video válido.")

        print(f"Modelo 3D generado en: {generated_3d_asset}")

        # Verificar que el archivo generado existe
        if not os.path.exists(generated_3d_asset):
            raise FileNotFoundError(f"El archivo 3D {generated_3d_asset} no existe.")

        # Paso 5: Extraer el archivo GLB
        print("Extrayendo archivo GLB...")
        result_extract_glb = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb"
        )
        extracted_glb_path = result_extract_glb[1]
        
        if not extracted_glb_path or not isinstance(extracted_glb_path, str):
            raise ValueError("Error al extraer GLB: respuesta inválida.")

        print(f"Archivo GLB extraído en: {extracted_glb_path}")

        # Verificar que el archivo GLB existe
        if not os.path.exists(extracted_glb_path):
            raise FileNotFoundError(f"El archivo GLB {extracted_glb_path} no existe.")

        # Finalizar sesión
        print("Finalizar sesión con la API...")
        client.predict(api_name="/end_session")
        
        # Paso 6: Subir archivos al almacenamiento
        print("Subiendo archivos al almacenamiento...")
        generation_folder = f'{user_uid}/TextoImagen3D/{generation_name}'
        generated_image_url = upload_to_storage(generated_image_path, f'{generation_folder}/generated_image.png')
        generated_3d_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/generated_3d.mp4')
        glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')

        # Paso 7: Guardar resultados en Firestore
        print("Guardando resultados en Firestore...")
        prediction_textimg3d_result = {
            "generation_name": generation_name,
            "prompt": prompt_generation,
            "generated_image": generated_image_url,
            "generated_3d": generated_3d_url,
            "glb_model_t23d": glb_url,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_type": "Texto a Imagen a 3D"
        }

        doc_ref = db.collection('predictions').document(user_uid).collection('TextoImagen3D').document(generation_name)
        doc_ref.set(prediction_textimg3d_result)
        
        print("Proceso completado con éxito.")
        return prediction_textimg3d_result

    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}")
        raise ValueError(str(e))

    except ValueError as e:
        print(f"Error de valor: {e}")
        raise ValueError(str(e))

    except Exception as e:
        error_message = str(e)
        if "You have exceeded your GPU quota" in error_message:
            raise ValueError("Has excedido el uso de GPU. Por favor, intenta más tarde.")
        elif "None" in error_message:
            raise ValueError("No existe GPU disponibles, inténtalo más tarde.")
        else:
            print(f"Error inesperado: {error_message}")
            raise ValueError(error_message)

    finally:
        # Eliminar archivos temporales solo si existen
        for path in [generated_image_path, preprocess_image_path, generated_3d_asset, extracted_glb_path]:
            if path and isinstance(path, str) and os.path.exists(path):
                os.remove(path)
                print(f"Archivo eliminado: {path}")

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