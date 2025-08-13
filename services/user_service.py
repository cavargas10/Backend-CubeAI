from config.firebase_config import db
import datetime
import firebase_admin
from firebase_admin import auth, firestore, storage
import logging

db = firestore.client()
bucket = storage.bucket()

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
        destination_blob_name = f"profile_pictures/{user_uid}"        
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
        logging.info(f"Foto de perfil actualizada para {user_uid}. Nueva URL: {cache_busting_url}")        
        return cache_busting_url
        
    except Exception as e:
        logging.error(f"Error en update_profile_picture: {str(e)}")
        raise

def delete_user(user_uid):
    user_ref = db.collection('users').document(user_uid)    
    try:
        blob_path = f"profile_pictures/{user_uid}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            blob.delete()
            logging.info(f"Foto de perfil eliminada de Storage para el usuario {user_uid}")
    except Exception as e:
        logging.error(f"No se pudo eliminar la foto de perfil de Storage para {user_uid}: {e}")

    user_ref.delete()
    auth.delete_user(user_uid)