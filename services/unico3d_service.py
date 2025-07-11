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

load_dotenv()

class Unico3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="Unico3D", readable_name="Unico a 3D")
        self.client = None 

    def _get_client(self):
        if self.client is None:
            print(f"Creando cliente Gradio para {self.collection_name}...")
            self.client = create_hf_client(os.getenv("CLIENT_UNICO3D_URL"))
            
    async def create_unico3d(self, user_uid, image_bytes, image_filename, generation_name):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        unique_filename = f"temp_image_unico_{uuid.uuid4().hex}_{image_filename}"
        with open(unique_filename, "wb") as f:
            f.write(image_bytes)
        
        temp_files_to_clean = [unique_filename]
        extracted_glb_path = None

        try:
            self._get_client() 
            loop = asyncio.get_running_loop()

            generate_func = partial(
                self.client.predict, 
                file(unique_filename),
                True,
                -1,
                False,
                True,
                0.1,
                "std",
                api_name="/generate3dv2" 
            )
            result_generate3dv2 = await loop.run_in_executor(None, generate_func)

            if isinstance(result_generate3dv2, tuple):
                extracted_glb_path = result_generate3dv2[0]
            else:
                extracted_glb_path = result_generate3dv2
            
            if not extracted_glb_path or not os.path.exists(extracted_glb_path):
                raise FileNotFoundError(f"El archivo GLB no se generó o no se encontró en la ruta: {extracted_glb_path}")
            temp_files_to_clean.append(extracted_glb_path)

            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
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

            return normalized_result
            
        except Exception as e:
            raise

        finally:
            for file_path in temp_files_to_clean:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error al eliminar el archivo temporal {file_path}: {e}")