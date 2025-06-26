import asyncio
import httpx
from functools import partial
from .base_generation_service import BaseGenerationService
from config.firebase_config import db
from config.huggingface_config import create_hf_client
from gradio_client import handle_file
from dotenv import load_dotenv
import datetime
import os
from utils.storage_utils import upload_to_storage
import tempfile

load_dotenv()

class TextImg3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="TextoImagen3D", readable_name="Texto a Imagen a 3D")
        self.client = None

    def _get_client(self):
        if self.client is None:
            print(f"Creando cliente Gradio para {self.collection_name}...")
            self.client = create_hf_client(os.getenv("CLIENT_TEXTOIMAGEN3D_URL"))

    async def create_2d_image(self, user_uid, generation_name, subject, style, additional_details):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        prompt_generation = f"{subject}, {additional_details}, style {style}, three quarter angle"
        generated_image_path = None

        try:
            self._get_client()
            loop = asyncio.get_running_loop()

            start_session_func = partial(self.client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            generate_image_func = partial(
                self.client.predict,
                prompt=prompt_generation,
                seed=42,
                randomize_seed=True,
                width=1024,
                height=1024,
                guidance_scale=3.5,
                api_name="/generate_flux_image"
            )
            generated_image_path = await loop.run_in_executor(None, generate_image_func)
            if not generated_image_path or not os.path.exists(generated_image_path):
                raise FileNotFoundError("Error al generar la imagen 2D base.")

            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
            image_url = upload_to_storage(generated_image_path, f'{generation_folder}/generated_2d_image.png')

            end_session_func = partial(self.client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)

            return {"generated_2d_image_url": image_url}

        finally:
            if generated_image_path and os.path.exists(generated_image_path):
                try:
                    os.remove(generated_image_path)
                except OSError as e:
                    print(f"Error al eliminar el archivo temporal {generated_image_path}: {e}")

    async def create_3d_from_image(self, user_uid, generation_name, image_url):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe.")

        temp_files_to_clean = []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(response.content)
                    downloaded_image_path = temp_file.name
            
            temp_files_to_clean.append(downloaded_image_path)

            self._get_client()
            loop = asyncio.get_running_loop()
            
            start_session_func = partial(self.client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            preprocess_func = partial(self.client.predict, image=handle_file(downloaded_image_path), api_name="/preprocess_image")
            preprocess_result = await loop.run_in_executor(None, preprocess_func)
            
            preprocess_image_path = preprocess_result[0] if isinstance(preprocess_result, (list, tuple)) else preprocess_result
            if not preprocess_image_path or not os.path.exists(preprocess_image_path):
                raise FileNotFoundError("Error al preprocesar la imagen.")
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
                raise FileNotFoundError("El archivo 3D generado no existe.")
            temp_files_to_clean.append(generated_3d_asset)

            extract_glb_func = partial(self.client.predict, mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")
            result_extract_glb = await loop.run_in_executor(None, extract_glb_func)
            extracted_glb_path = result_extract_glb[1]
            if not os.path.exists(extracted_glb_path):
                raise FileNotFoundError("El archivo GLB extraído no existe.")
            temp_files_to_clean.append(extracted_glb_path)

            end_session_func = partial(self.client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)
            
            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
            glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')
            preview_video_url = upload_to_storage(generated_3d_asset, f'{generation_folder}/preview.mp4')

            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": glb_url,
                "previewUrl": preview_video_url,
                "downloads": [{"format": "GLB", "url": glb_url}],
                "raw_data": {
                    "generated_2d_image_url": image_url
                }
            }

            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)

            return normalized_result

        finally:
            for file_path in temp_files_to_clean:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error al eliminar el archivo temporal {file_path}: {e}")