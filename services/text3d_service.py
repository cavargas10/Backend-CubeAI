import asyncio
from functools import partial
from .base_generation_service import BaseGenerationService
from config.firebase_config import db
from config.huggingface_config import create_hf_client
from dotenv import load_dotenv
import datetime
import os
from utils.storage_utils import upload_to_storage
import logging

load_dotenv()

class Text3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="Texto3D", readable_name="Texto a 3D")
        self.gradio_url = os.getenv("CLIENT_TEXTO3D_URL")

    async def create_text3d(self, user_uid, generation_name, user_prompt, selected_style):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        style_keywords = {
            "realistic": "photorealistic, 8k, hyper-detailed, octane render, cinematic lighting, ultra-realistic",
            "disney": "disney pixar style, friendly character, vibrant colors, smooth shading, 3d animation movie style",
            "anime": "anime key visual, studio ghibli style, cel shaded, japanese animation, detailed character design",
            "chibi": "chibi style, cute, big expressive eyes, small body, kawaii, miniature",
            "pixar": "pixar movie style, detailed textures, expressive character, 3d animated film scene",
        }

        logging.info(f"Estilo seleccionado recibido: '{selected_style}'")
        
        if selected_style and selected_style in style_keywords:
            logging.info(f"Aplicando aumento de prompt para el estilo: {selected_style}")
            selected_keywords = style_keywords[selected_style]
            prompt_final = f"{selected_style} style, {selected_keywords}. A 3D model of: {user_prompt}. Detailed, high quality."
        else:
            logging.info("No se aplicó un estilo predefinido. Usando el prompt del usuario directamente.")
            prompt_final = f"A detailed 3D model of: {user_prompt}. high quality, sharp focus."

        logging.info(f"Prompt final enviado a la API: '{prompt_final}'")
        
        temp_files_to_clean = []
        client = None

        try:
            logging.info(f"Creando una nueva instancia de cliente Gradio para el trabajo {generation_name}.")
            client = create_hf_client(self.gradio_url)
            loop = asyncio.get_running_loop()

            start_session_func = partial(client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            get_seed_func = partial(client.predict, randomize_seed=True, seed=0, api_name="/get_seed")
            seed_value = await loop.run_in_executor(None, get_seed_func)
            if not isinstance(seed_value, int):
                raise ValueError(f"Seed inválido: {seed_value}")

            text_to_3d_func = partial(
                client.predict,
                prompt=prompt_final,
                seed=seed_value,
                ss_guidance_strength=7.5,
                ss_sampling_steps=25,
                slat_guidance_strength=7.5,
                slat_sampling_steps=25,
                api_name="/text_to_3d"
            )
            result_text_to_3d = await loop.run_in_executor(None, text_to_3d_func)
            if not isinstance(result_text_to_3d, dict) or "video" not in result_text_to_3d:
                raise ValueError(f"Error al generar modelo 3D: respuesta de la API inválida: {result_text_to_3d}")

            generated_video_path = result_text_to_3d["video"]
            if not generated_video_path or not os.path.exists(generated_video_path):
                 raise FileNotFoundError(f"El archivo de video generado no se encontró. Respuesta de la API: {generated_video_path}")
            temp_files_to_clean.append(generated_video_path)

            extract_glb_func = partial(client.predict, mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")
            result_extract_glb = await loop.run_in_executor(None, extract_glb_func)
            if not result_extract_glb or not isinstance(result_extract_glb, (list, tuple)) or len(result_extract_glb) < 2:
                raise ValueError(f"Respuesta inesperada de 'extract_glb': {result_extract_glb}")

            extracted_glb_path = result_extract_glb[1]
            if not extracted_glb_path or not os.path.exists(extracted_glb_path):
                 raise FileNotFoundError(f"El archivo GLB extraído no se encontró. Respuesta de la API: {extracted_glb_path}")
            temp_files_to_clean.append(extracted_glb_path)

            end_session_func = partial(client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)
            
            generation_folder = f'users/{user_uid}/generations/{self.collection_name}/{generation_name}'
            glb_url = upload_to_storage(extracted_glb_path, f'{generation_folder}/model.glb')

            normalized_result = {
                "generation_name": generation_name,
                "prediction_type": self.readable_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "modelUrl": glb_url,
                "downloads": [{"format": "GLB", "url": glb_url}],
                "raw_data": {
                    "user_prompt": user_prompt,
                    "selected_style": selected_style,
                    "full_prompt_sent_to_api": prompt_final,
                }
            }

            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)

            return normalized_result

        except Exception as e:
            logging.error(f"Excepción en create_text3d para {generation_name}: {e}", exc_info=True)
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