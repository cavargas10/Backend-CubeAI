from config.firebase_config import db, bucket
from utils.storage_utils import upload_to_storage
from flask import current_app

class BaseGenerationService:
    def __init__(self, collection_name: str, readable_name: str):
        if not collection_name or not readable_name:
            raise ValueError("El nombre de la colección y el nombre legible son requeridos.")
        self.collection_name = collection_name
        self.readable_name = readable_name

    def _generation_exists(self, user_uid: str, generation_name: str) -> bool:
        doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
        return doc_ref.get().exists

    def get_generations(self, user_uid: str) -> list:
        generations_ref = db.collection('predictions').document(user_uid).collection(self.collection_name)
        return [gen.to_dict() for gen in generations_ref.stream()]

    def add_preview_image(self, user_uid: str, generation_name: str, preview_file) -> dict:
        doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
        doc = doc_ref.get()

        if not doc.exists:
            raise ValueError(f"No se encontró la generación '{generation_name}' para el usuario.")

        generation_folder = f'users/{user_uid}/generations/{self.collection_name}/{generation_name}'
        
        try:
            preview_image_url = upload_to_storage(preview_file, f'{generation_folder}/preview_image.png')
            update_data = {"previewImageUrl": preview_image_url}
            doc_ref.update(update_data)

            updated_doc_data = doc.to_dict()
            updated_doc_data.update(update_data)
            return updated_doc_data
        except Exception as e:
            current_app.logger.error(f"Error al subir preview para {generation_name}: {e}", exc_info=True)
            raise 

    def delete_generation(self, user_uid: str, generation_name: str) -> bool:
        doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
        doc = doc_ref.get()

        if not doc.exists:
            return False

        generation_folder = f"users/{user_uid}/generations/{self.collection_name}/{generation_name}"
        
        try:
            blobs = bucket.list_blobs(prefix=generation_folder)
            for blob in blobs:
                blob.delete()
        except Exception as e:
            current_app.logger.error(f"Error al eliminar archivos de Storage para {generation_name}: {e}", exc_info=True)

        doc_ref.delete()
        return True