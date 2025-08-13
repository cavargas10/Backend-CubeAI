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

class Retexturize3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="Retexturize3D", readable_name="Estudio de Texturizado")
        self.gradio_url = os.getenv("CLIENT_RETEXTURE3D_URL")
        if not self.gradio_url:
            raise ValueError("La URL del cliente de Gradio para Retexturize3D no está configurada en .env")

    async def create_retexture3d(self, user_uid, generation_name, model_bytes, model_filename, texture_bytes, texture_filename):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        temp_model_filename = f"temp_model_{uuid.uuid4().hex}_{model_filename}"
        temp_texture_filename = f"temp_texture_{uuid.uuid4().hex}_{texture_filename}"
        
        with open(temp_model_filename, "wb") as f:
            f.write(model_bytes)
        with open(temp_texture_filename, "wb") as f:
            f.write(texture_bytes)
            
        temp_files_to_clean = [temp_model_filename, temp_texture_filename]
        client = None

        try:
            logging.info(f"Creando cliente Gradio para trabajo de retexturizado: {generation_name}")
            client = create_hf_client(self.gradio_url)
            loop = asyncio.get_running_loop()

            start_session_func = partial(client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)
            logging.info(f"Sesión iniciada en Gradio para {generation_name}.")

            get_seed_func = partial(client.predict, randomize_seed=True, seed=0, api_name="/get_random_seed")
            seed_value = await loop.run_in_executor(None, get_seed_func)
            if not isinstance(seed_value, int):
                raise ValueError(f"Seed inválido recibido de la API: {seed_value}")
            logging.info(f"Seed obtenido para {generation_name}: {seed_value}")

            generate_texture_func = partial(
                client.predict,
                input_image_path=handle_file(temp_texture_filename),
                input_mesh_path=handle_file(temp_model_filename),
                guidance_scale=3,
                inference_steps=50,
                seed=seed_value,
                reference_conditioning_scale=1,
                api_name="/generate_texture"
            )
            logging.info(f"Iniciando generación de textura para {generation_name}.")
            result_path = await loop.run_in_executor(None, generate_texture_func)
            
            if not result_path or not os.path.exists(result_path):
                raise FileNotFoundError(f"El archivo del modelo retexturizado no se encontró. Respuesta de API: {result_path}")
            
            temp_files_to_clean.append(result_path)
            logging.info(f"Textura generada exitosamente para {generation_name}. Archivo en: {result_path}")

            end_session_func = partial(client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)
            logging.info(f"Sesión finalizada en Gradio para {generation_name}.")

            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
            retextured_model_url = upload_to_storage(result_path, f'{generation_folder}/model.glb')
            original_model_url = upload_to_storage(temp_model_filename, f'{generation_folder}/original_model.glb')
            texture_image_url = upload_to_storage(temp_texture_filename, f'{generation_folder}/texture_reference.png')
            
            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": retextured_model_url,
                "downloads": [{"format": "GLB", "url": retextured_model_url}],
                "raw_data": {
                    "original_model_url": original_model_url,
                    "texture_image_url": texture_image_url,
                }
            }

            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)
            
            logging.info(f"Trabajo {generation_name} completado y guardado en Firestore.")
            return normalized_result
        
        except Exception as e:
            logging.error(f"Excepción en create_retexture3d para {generation_name}: {e}", exc_info=True)
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
                    logging.warning(f"Error al cerrar cliente Gradio para {generation_name}: {e}")