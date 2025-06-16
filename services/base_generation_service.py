from config.firebase_config import db, bucket
from utils.storage_utils import upload_to_storage
from flask import current_app

class BaseGenerationService:
    """
    Clase base para los servicios de generación.
    Contiene la lógica común para interactuar con Firestore y Storage.
    """
    def __init__(self, collection_name: str, readable_name: str):
        """
        Inicializa el servicio.
        Args:
            collection_name (str): El nombre de la colección en Firestore (ej. 'Imagen3D').
            readable_name (str): El nombre legible para el tipo de predicción (ej. 'Imagen a 3D').
        """
        if not collection_name or not readable_name:
            raise ValueError("El nombre de la colección y el nombre legible son requeridos.")
        self.collection_name = collection_name
        self.readable_name = readable_name

    def _generation_exists(self, user_uid: str, generation_name: str) -> bool:
        """Verifica si una generación ya existe en Firestore."""
        doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
        return doc_ref.get().exists

    def get_generations(self, user_uid: str) -> list:
        """Obtiene todas las generaciones de este tipo para un usuario."""
        generations_ref = db.collection('predictions').document(user_uid).collection(self.collection_name)
        return [gen.to_dict() for gen in generations_ref.stream()]

    def add_preview_image(self, user_uid: str, generation_name: str, preview_file) -> dict:
        """Añade una imagen de previsualización a una generación existente."""
        doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
        doc = doc_ref.get()

        if not doc.exists:
            raise ValueError(f"No se encontró la generación '{generation_name}' para el usuario.")

        generation_folder = f'{user_uid}/{self.collection_name}/{generation_name}'
        try:
            preview_image_url = upload_to_storage(preview_file, f'{generation_folder}/preview_image.png')
            update_data = {"previewImageUrl": preview_image_url}
            doc_ref.update(update_data)

            # Devolver el documento actualizado
            updated_doc_data = doc.to_dict()
            updated_doc_data.update(update_data)
            return updated_doc_data
        except Exception as e:
            current_app.logger.error(f"Error al subir preview para {generation_name}: {e}", exc_info=True)
            raise # Re-lanza la excepción para que la ruta la maneje

    def delete_generation(self, user_uid: str, generation_name: str) -> bool:
        """Elimina una generación y todos sus archivos asociados de Storage."""
        doc_ref = db.collection('predictions').document(user_uid).collection(self.collection_name).document(generation_name)
        doc = doc_ref.get()

        if not doc.exists:
            return False

        # Eliminar archivos de Firebase Storage
        generation_folder = f"{user_uid}/{self.collection_name}/{generation_name}"
        try:
            blobs = bucket.list_blobs(prefix=generation_folder)
            for blob in blobs:
                blob.delete()
        except Exception as e:
            current_app.logger.error(f"Error al eliminar archivos de Storage para {generation_name}: {e}", exc_info=True)

        # Eliminar documento de Firestore
        doc_ref.delete()
        return True