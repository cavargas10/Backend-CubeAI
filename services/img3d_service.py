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

load_dotenv()

class Img3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="Imagen3D", readable_name="Imagen a 3D")
        self.client = None 

    def _get_client(self):
        if self.client is None:
            print(f"Creando cliente Gradio para {self.collection_name}...")
            self.client = create_hf_client(os.getenv("CLIENT_IMAGEN3D_URL"))
        return self.client

    async def create_generation(self, user_uid, image_bytes, image_filename, generation_name):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        unique_filename = f"temp_image_{uuid.uuid4().hex}_{image_filename}"
        with open(unique_filename, "wb") as f:
            f.write(image_bytes)
        
        temp_files_to_clean = [unique_filename]

        try:
            self._get_client()
            loop = asyncio.get_running_loop()

            start_session_func = partial(self.client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            preprocess_func = partial(self.client.predict, image=handle_file(unique_filename), api_name="/preprocess_image")
            preprocess_image_path = await loop.run_in_executor(None, preprocess_func)
            
            if not os.path.exists(preprocess_image_path):
                raise FileNotFoundError(f"El archivo preprocesado {preprocess_image_path} no existe.")
            temp_files_to_clean.append(preprocess_image_path)

            get_seed_func = partial(self.client.predict, randomize_seed=True, seed=0, api_name="/get_seed")
            seed_value = await loop.run_in_executor(None, get_seed_func)
            
            image_to_3d_func = partial(
                self.client.predict,
                image=handle_file(preprocess_image_path),
                seed=seed_value,
                ss_guidance_strength=7.5,
                ss_sampling_steps=12,
                slat_guidance_strength=3,
                slat_sampling_steps=12,
                api_name="/image_to_3d"
            )
            result_image_to_3d = await loop.run_in_executor(None, image_to_3d_func)
            generated_3d_asset = result_image_to_3d["video"]
            if not os.path.exists(generated_3d_asset):
                raise FileNotFoundError(f"El archivo generado {generated_3d_asset} no existe.")
            temp_files_to_clean.append(generated_3d_asset)

            extract_glb_func = partial(self.client.predict, mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")
            result_extract_glb = await loop.run_in_executor(None, extract_glb_func)
            extracted_glb_path = result_extract_glb[1]
            
            if not os.path.exists(extracted_glb_path):
                raise FileNotFoundError(f"El archivo GLB {extracted_glb_path} no existe.")
            temp_files_to_clean.append(extracted_glb_path)
            
            end_session_func = partial(self.client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)

            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
            glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')        
            preview_video_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/preview.mp4')       
            preprocess_url = upload_to_storage(preprocess_image_path, f'{generation_folder}/preprocess.png')
            
            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": glb_url,
                "previewUrl": preview_video_url,
                "downloads": [{"format": "GLB", "url": glb_url}],
                "raw_data": {"preprocess_image_url": preprocess_url}
            }
            
            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)
            
            return normalized_result
        
        except Exception as e:
            error_message = str(e)
            if "You have exceeded your GPU quota" in error_message:
                raise ValueError("Has excedido tu cuota de uso de GPU. Por favor, intenta más tarde.")
            elif "None" in error_message: 
                raise ValueError("No hay GPUs disponibles en este momento, por favor inténtalo más tarde.")
            else:
                raise
        
        finally:
            for file_path in temp_files_to_clean:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error al eliminar el archivo temporal {file_path}: {e}")