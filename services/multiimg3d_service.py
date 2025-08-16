import asyncio
from functools import partial
from .base_generation_service import BaseGenerationService
from config.firebase_config import db
from config.huggingface_config import create_hf_client
from gradio_client import handle_file
from dotenv import load_dotenv
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage
import logging

load_dotenv()

class MultiImg3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="MultiImagen3D", readable_name="Multi Imagen a 3D")
        self.gradio_url = os.getenv("CLIENT_MULTI3D_URL")

    async def create_multiimg3d(self, user_uid, frontal_bytes, lateral_bytes, trasera_bytes, generation_name):

        temp_input_files = {
            "frontal": f"temp_frontal_{uuid.uuid4().hex}.png",
            "lateral": f"temp_lateral_{uuid.uuid4().hex}.png",
            "trasera": f"temp_trasera_{uuid.uuid4().hex}.png"
        }
        
        with open(temp_input_files["frontal"], "wb") as f: f.write(frontal_bytes)
        with open(temp_input_files["lateral"], "wb") as f: f.write(lateral_bytes)
        with open(temp_input_files["trasera"], "wb") as f: f.write(trasera_bytes)

        temp_files_to_clean = list(temp_input_files.values())
        client = None

        try:
            logging.info(f"Creando una nueva instancia de cliente Gradio para el trabajo {generation_name}.")
            client = create_hf_client(self.gradio_url)
            loop = asyncio.get_running_loop()

            start_session_func = partial(client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            preprocess_func = partial(
                client.predict,
                images=[
                    {"image": handle_file(temp_input_files["frontal"])},
                    {"image": handle_file(temp_input_files["lateral"])},
                    {"image": handle_file(temp_input_files["trasera"])}
                ],
                api_name="/preprocess_images"
            )
            
            logging.info(f"Enviando imágenes para preprocesamiento para el trabajo {generation_name}.")
            preprocess_results = await loop.run_in_executor(None, preprocess_func)
            
            if not isinstance(preprocess_results, list) or len(preprocess_results) != 3:
                raise ValueError(f"Error al preprocesar las imágenes: se esperaban 3 imágenes, se obtuvieron {len(preprocess_results) if isinstance(preprocess_results, list) else 'respuesta no válida'}.")

            preprocess_paths = [res["image"] for res in preprocess_results]
            for path in preprocess_paths:
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"Un archivo preprocesado no se encontró. Respuesta de la API: {preprocess_paths}")
            temp_files_to_clean.extend(preprocess_paths)
            logging.info(f"Preprocesamiento de imágenes completado para el trabajo {generation_name}.")

            get_seed_func = partial(client.predict, randomize_seed=True, seed=0, api_name="/get_seed")
            seed_value = await loop.run_in_executor(None, get_seed_func)
            if not isinstance(seed_value, int):
                raise ValueError(f"Seed inválido: {seed_value}")

            image_to_3d_func = partial(
                client.predict,
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
            
            logging.info(f"Iniciando generación 3D para el trabajo {generation_name}.")
            result_image_to_3d = await loop.run_in_executor(None, image_to_3d_func)
            if not isinstance(result_image_to_3d, dict) or "video" not in result_image_to_3d:
                raise ValueError(f"Error al generar modelo 3D: respuesta inválida: {result_image_to_3d}")
            
            generated_3d_asset = result_image_to_3d["video"]
            if not generated_3d_asset or not os.path.exists(generated_3d_asset):
                raise FileNotFoundError(f"El archivo 3D generado no se encontró. Respuesta de la API: {generated_3d_asset}")
            temp_files_to_clean.append(generated_3d_asset)

            extract_glb_func = partial(client.predict, mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")
            
            logging.info(f"Extrayendo GLB para el trabajo {generation_name}.")
            result_extract_glb = await loop.run_in_executor(None, extract_glb_func)
            if not result_extract_glb or not isinstance(result_extract_glb, (list, tuple)) or len(result_extract_glb) < 2:
                raise ValueError(f"Respuesta inesperada de 'extract_glb': {result_extract_glb}")

            extracted_glb_path = result_extract_glb[1]
            if not extracted_glb_path or not os.path.exists(extracted_glb_path):
                raise FileNotFoundError(f"El archivo GLB extraído no se encontró. Respuesta de la API: {extracted_glb_path}")
            temp_files_to_clean.append(extracted_glb_path)

            end_session_func = partial(client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)
            logging.info(f"Sesión de Gradio finalizada para {generation_name}.")

            generation_folder = f'users/{user_uid}/generations/{self.collection_name}/{generation_name}'
            
            glb_url_base = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
            input_urls = {
                "frontal": upload_to_storage(temp_input_files["frontal"], f'{generation_folder}/input_frontal.png'),
                "lateral": upload_to_storage(temp_input_files["lateral"], f'{generation_folder}/input_lateral.png'),
                "trasera": upload_to_storage(temp_input_files["trasera"], f'{generation_folder}/input_trasera.png')
            }

            timestamp_query = f"?v={int(datetime.datetime.now().timestamp())}"
            glb_url_with_cache_buster = f"{glb_url_base}{timestamp_query}"

            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": glb_url_with_cache_buster,
                "downloads": [{"format": "GLB", "url": glb_url_with_cache_buster}],
                "raw_data": {"input_image_urls": input_urls}
            }

            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)

            logging.info(f"Trabajo {generation_name} completado y guardado en Firestore.")
            return normalized_result

        except Exception as e:
            logging.error(f"Excepción en create_multiimg3d para {generation_name}: {e}", exc_info=True)
            raise

        finally:
            for file_path in temp_files_to_clean:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        logging.warning(f"No se pudo eliminar el archivo temporal {file_path}: {e}")
            
            if client:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, client.close)
                    logging.info(f"Cliente Gradio para {generation_name} cerrado.")
                except Exception as e:
                    logging.warning(f"Error al cerrar el cliente Gradio para {generation_name}: {e}")