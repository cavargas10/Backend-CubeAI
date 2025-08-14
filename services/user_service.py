from config.firebase_config import db, bucket
import datetime
from firebase_admin import auth
import logging

def register_user(user_data):
    user_ref = db.collection('users').document(user_data["uid"])
    user_ref.set({
        "email": user_data["email"],
        "name": user_data["name"],
        "profile_picture": user_data.get("profile_picture", ""),
        "created_at": datetime.datetime.now()
    }, merge=True)

def get_user_data(user_uid):
    user_ref = db.collection('users').document(user_uid)
    user_doc = user_ref.get()
    return user_doc.to_dict() if user_doc.exists else None

def update_user_name(user_uid, new_name):
    try:
        user_ref = db.collection('users').document(user_uid)
        user_ref.update({"name": new_name})
        return get_user_data(user_uid)
    except Exception as e:
        logging.error(f"Error en update_user_name: {str(e)}")
        raise

def update_profile_picture(user_uid, uploaded_file):
    try:
        destination_blob_name = f"users/{user_uid}/profile_picture/image"        
        blob = bucket.blob(destination_blob_name)        
        uploaded_file.file.seek(0)        
        blob.upload_from_file(
            uploaded_file.file,
            content_type=uploaded_file.content_type
        )
        
        blob.make_public()
        base_url = blob.public_url
        cache_busting_url = f"{base_url}?updated={int(datetime.datetime.now().timestamp())}"
        
        user_ref = db.collection('users').document(user_uid)
        user_ref.update({"profile_picture": cache_busting_url})
        logging.info(f"Foto de perfil actualizada para {user_uid} en {destination_blob_name}")
        return cache_busting_url
        
    except Exception as e:
        logging.error(f"Error en update_profile_picture: {str(e)}")
        raise

def delete_user(user_uid):
    user_ref = db.collection('users').document(user_uid)
    user_ref.delete()

    try:
        user_folder_prefix = f"users/{user_uid}/"
        blobs_to_delete = bucket.list_blobs(prefix=user_folder_prefix)
        for blob in blobs_to_delete:
            blob.delete()
        logging.info(f"Todos los archivos de Storage para el usuario {user_uid} han sido eliminados.")
    except Exception as e:
        logging.error(f"No se pudieron eliminar los archivos de Storage para {user_uid}: {e}")
    auth.delete_user(user_uid)