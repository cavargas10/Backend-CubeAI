import asyncio
from functools import partial
from .base_generation_service import BaseGenerationService
from config.firebase_config import db
from gradio_client import file
from config.huggingface_config import create_hf_client
from dotenv import load_dotenv
import datetime
import uuid
import os
from utils.storage_utils import upload_to_storage
import logging

load_dotenv()

class Unico3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="Unico3D", readable_name="Unico a 3D")
        self.gradio_url = os.getenv("CLIENT_UNICO3D_URL")

    async def create_unico3d(self, user_uid, image_bytes, image_filename, generation_name):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        unique_filename = f"temp_image_unico_{uuid.uuid4().hex}_{image_filename}"
        with open(unique_filename, "wb") as f:
            f.write(image_bytes)
        
        temp_files_to_clean = [unique_filename]
        client = None

        try:
            logging.info(f"Creando una nueva instancia de cliente Gradio para el trabajo {generation_name}.")
            client = create_hf_client(self.gradio_url)
            loop = asyncio.get_running_loop()

            generate_func = partial(
                client.predict, 
                file(unique_filename),  
                True,  # render_video 
                -1,    # seed
                False, # do_refine
                True,  # input_processing
                0.1,   # expansion_weight
                "std", # init_type
                api_name="/generate3dv2" 
            )
            
            logging.info(f"Enviando trabajo {generation_name} al Space Unique3D...")
            result_generate3dv2 = await loop.run_in_executor(None, generate_func)
            logging.info(f"Respuesta recibida del Space para {generation_name}.")

            if isinstance(result_generate3dv2, tuple) and len(result_generate3dv2) > 0:
                extracted_glb_path = result_generate3dv2[0]
            elif isinstance(result_generate3dv2, str):
                extracted_glb_path = result_generate3dv2
            else:
                raise ValueError(f"Respuesta inesperada de la API de Unique3D: {result_generate3dv2}")
            
            if not extracted_glb_path or not os.path.exists(extracted_glb_path):
                raise FileNotFoundError(f"El archivo GLB no se generó o no se encontró en la ruta devuelta: {extracted_glb_path}")

            temp_files_to_clean.append(extracted_glb_path)

            generation_folder = f'users/{user_uid}/generations/{self.collection_name}/{generation_name}'
            glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
            
            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": glb_url,
                "previewUrl": None,
                "downloads": [{"format": "GLB", "url": glb_url}],
                "raw_data": {}
            }

            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)

            logging.info(f"Trabajo {generation_name} completado y guardado en Firestore.")
            return normalized_result
            
        except Exception as e:
            logging.error(f"Excepción en create_unico3d para {generation_name}: {e}", exc_info=True)
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