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
import logging

load_dotenv()
class TextImg3DService(BaseGenerationService):
    def __init__(self):
        super().__init__(collection_name="TextoImagen3D", readable_name="Texto a Imagen a 3D")
        self.gradio_url = os.getenv("CLIENT_TEXTOIMAGEN3D_URL")

    async def create_2d_image(self, user_uid, generation_name, prompt, selected_style):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe. Por favor, elige otro nombre.")

        style_keywords = {
            "realistic": "photorealistic, 8k, hyper-detailed, octane render, cinematic lighting, ultra-realistic",
            "disney": "disney pixar style, friendly character, vibrant colors, smooth shading, 3d animation movie style",
            "anime": "anime key visual, studio ghibli style, cel shaded, japanese animation, detailed character design",
            "chibi": "chibi style, cute, big expressive eyes, small body, kawaii, miniature",
            "pixar": "pixar movie style, detailed textures, expressive character, 3d animated film scene",
        }

        logging.info(f"Estilo seleccionado para imagen 2D: '{selected_style}'")
        
        if selected_style and selected_style in style_keywords:
            logging.info(f"Aplicando aumento de prompt para el estilo 2D: {selected_style}")
            selected_keywords = style_keywords[selected_style]
            prompt_final = f"award-winning photo of {prompt}, {selected_keywords}, ({selected_style} style)"
        else:
            logging.info("No se aplicó un estilo predefinido. Usando el prompt del usuario para imagen 2D.")
            prompt_final = f"award-winning photo of {prompt}, 4k, detailed"

        logging.info(f"Prompt final para 2D enviado a la API: '{prompt_final}'")
        
        generated_image_path = None
        client = None

        try:
            logging.info(f"Creando una nueva instancia de cliente Gradio para el trabajo 2D {generation_name}.")
            client = create_hf_client(self.gradio_url)
            loop = asyncio.get_running_loop()

            start_session_func = partial(client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            generate_image_func = partial(
                client.predict,
                prompt=prompt_final,
                seed=42,
                randomize_seed=True,
                width=1024,
                height=1024,
                guidance_scale=3.5,
                api_name="/generate_flux_image"
            )

            logging.info(f"Iniciando generación de imagen 2D para el trabajo {generation_name}.")
            generated_image_path = await loop.run_in_executor(None, generate_image_func)
            
            if not generated_image_path or not os.path.exists(generated_image_path):
                raise FileNotFoundError(f"Error al generar la imagen 2D. No se encontró el archivo. Respuesta de la API: {generated_image_path}")

            logging.info(f"Imagen 2D generada para {generation_name}. Subiendo a storage...")
            generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
            image_url = upload_to_storage(generated_image_path, f'{generation_folder}/generated_2d_image.png')

            end_session_func = partial(client.predict, api_name="/end_session")
            await loop.run_in_executor(None, end_session_func)

            return {"generated_2d_image_url": image_url}

        except Exception as e:
            logging.error(f"Excepción en create_2d_image para {generation_name}: {e}", exc_info=True)
            raise

        finally:
            if generated_image_path and os.path.exists(generated_image_path):
                try:
                    os.remove(generated_image_path)
                except OSError as e:
                    logging.warning(f"No se pudo eliminar el archivo temporal de imagen 2D {generated_image_path}: {e}")
            if client:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, client.close)
                    logging.info(f"Cliente Gradio para el trabajo 2D {generation_name} cerrado.")
                except Exception as e:
                    logging.warning(f"Error al cerrar el cliente Gradio para el trabajo 2D {generation_name}: {e}")

    async def create_3d_from_image(self, user_uid, generation_name, image_url):
        if self._generation_exists(user_uid, generation_name):
            raise ValueError("El nombre de la generación ya existe.")

        temp_files_to_clean = []
        client = None
        
        try:
            logging.info(f"Descargando imagen 2D de {image_url} para el trabajo 3D {generation_name}.")
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(image_url, timeout=60.0)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(response.content)
                    downloaded_image_path = temp_file.name
            
            temp_files_to_clean.append(downloaded_image_path)
            logging.info(f"Imagen 2D descargada en: {downloaded_image_path}")

            logging.info(f"Creando una nueva instancia de cliente Gradio para el trabajo 3D {generation_name}.")
            client = create_hf_client(self.gradio_url)
            loop = asyncio.get_running_loop()
            
            start_session_func = partial(client.predict, api_name="/start_session")
            await loop.run_in_executor(None, start_session_func)

            preprocess_func = partial(client.predict, image=handle_file(downloaded_image_path), api_name="/preprocess_image")
            preprocess_result = await loop.run_in_executor(None, preprocess_func)
            
            preprocess_image_path = preprocess_result[0] if isinstance(preprocess_result, (list, tuple)) else preprocess_result
            if not preprocess_image_path or not os.path.exists(preprocess_image_path):
                raise FileNotFoundError(f"Error al preprocesar la imagen. Respuesta de la API: {preprocess_image_path}")
            temp_files_to_clean.append(preprocess_image_path)
            
            get_seed_func = partial(client.predict, randomize_seed=True, seed=0, api_name="/get_seed")
            seed_value = await loop.run_in_executor(None, get_seed_func)

            image_to_3d_func = partial(
                client.predict,
                image=handle_file(preprocess_image_path),
                seed=seed_value,
                ss_guidance_strength=7.5,
                ss_sampling_steps=12,
                slat_guidance_strength=3,
                slat_sampling_steps=12,
                api_name="/image_to_3d"
            )
            result_image_to_3d = await loop.run_in_executor(None, image_to_3d_func)
            if not isinstance(result_image_to_3d, dict) or "video" not in result_image_to_3d:
                raise ValueError(f"Respuesta inválida de image_to_3d: {result_image_to_3d}")

            generated_3d_asset = result_image_to_3d["video"]
            if not generated_3d_asset or not os.path.exists(generated_3d_asset):
                raise FileNotFoundError(f"El archivo 3D generado no se encontró. Respuesta de la API: {generated_3d_asset}")
            temp_files_to_clean.append(generated_3d_asset)

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
                "raw_data": {"generated_2d_image_url": image_url}
            }

            doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
            doc_ref.set(normalized_result)

            logging.info(f"Trabajo 3D {generation_name} completado y guardado en Firestore.")
            return normalized_result

        except httpx.HTTPStatusError as e:
            logging.error(f"Error HTTP al descargar la imagen para {generation_name}: {e.response.status_code}", exc_info=True)
            raise ValueError(f"No se pudo descargar la imagen 2D desde la URL. Código de estado: {e.response.status_code}")
        except Exception as e:
            logging.error(f"Excepción en create_3d_from_image para {generation_name}: {e}", exc_info=True)
            raise

        finally:
            for file_path in temp_files_to_clean:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        logging.warning(f"No se pudo eliminar el archivo temporal 3D {file_path}: {e}")
            if client:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, client.close)
                    logging.info(f"Cliente Gradio para el trabajo 3D {generation_name} cerrado.")
                except Exception as e:
                    logging.warning(f"Error al cerrar el cliente Gradio para el trabajo 3D {generation_name}: {e}")