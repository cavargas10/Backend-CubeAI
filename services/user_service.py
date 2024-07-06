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

def update_profile_picture(user_uid, file):
    try:
        # Generate a unique filename
        filename = f"{user_uid}_{datetime.datetime.now().timestamp()}"
        
        # Upload file to Firebase Storage
        blob = bucket.blob(filename)
        blob.upload_from_file(file)
        
        # Make the blob publicly accessible
        blob.make_public()
        
        # Get the public URL
        url = blob.public_url
        
        # Update user document with new profile picture URL
        user_ref = db.collection('users').document(user_uid)
        user_ref.update({"profile_picture": url})
        
        return url
    except Exception as e:
        logging.error(f"Error en update_profile_picture: {str(e)}")
        raise

def delete_user(user_uid):
    user_ref = db.collection('users').document(user_uid)
    user_ref.delete()
    auth.delete_user(user_uid)