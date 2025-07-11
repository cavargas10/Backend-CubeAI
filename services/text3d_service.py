import asyncio
from functools import partial
from .base_generation_service import BaseGenerationService
from config.firebase_config import db
from config.huggingface_config import create_hf_client
from dotenv import load_dotenv
import datetime
import os
from utils.storage_utils import upload_to_storage

load_dotenv()

class Text3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="Texto3D", readable_name="Texto a 3D")
        self.client = None 

    def _get_client(self):
        if self.client is None:
            print(f"Creando cliente Gradio para {self.collection_name}...")
            self.client = create_hf_client(os.getenv("CLIENT_TEXTO3D_URL"))
        return self.client

    async def create_text3d(self, user_uid, generation_name, user_prompt, selected_style):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        full_prompt = f"A {selected_style} 3D render of {user_prompt}. Style: {selected_style}. Emphasize essential features and textures with vibrant colors."
        temp_files_to_clean = []

        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()

            start_session_func = partial(client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            get_seed_func = partial(client.predict, randomize_seed=True, seed=0, api_name="/get_seed")
            result_get_seed = await loop.run_in_executor(None, get_seed_func)
            if not isinstance(result_get_seed, int):
                raise ValueError(f"Seed inválido: {result_get_seed}")
            seed_value = result_get_seed

            text_to_3d_func = partial(
                client.predict,
                prompt=full_prompt,
                seed=seed_value,
                ss_guidance_strength=7.5,
                ss_sampling_steps=25,
                slat_guidance_strength=7.5,
                slat_sampling_steps=25,
                api_name="/text_to_3d"
            )
            result_text_to_3d = await loop.run_in_executor(None, text_to_3d_func)
            if not isinstance(result_text_to_3d, dict) or "video" not in result_text_to_3d:
                raise ValueError("Error al generar modelo 3D: respuesta de la API inválida.")

            generated_video_path = result_text_to_3d["video"]
            temp_files_to_clean.append(generated_video_path)

            extract_glb_func = partial(client.predict, mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")
            result_extract_glb = await loop.run_in_executor(None, extract_glb_func)
            extracted_glb_path = result_extract_glb[1]
            temp_files_to_clean.append(extracted_glb_path)

            end_session_func = partial(client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)
            
            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
            glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
            preview_video_url = upload_to_storage(generated_video_path, f'{generation_folder}/preview.mp4')

            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": glb_url,
                "previewUrl": preview_video_url,
                "downloads": [{"format": "GLB", "url": glb_url}],
                "raw_data": {
                    "user_prompt": user_prompt,
                    "selected_style": selected_style,
                    "full_prompt_sent_to_api": full_prompt,
                }
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