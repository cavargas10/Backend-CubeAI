import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, storage

load_dotenv()

cred_path = os.getenv('FIREBASE_CREDENTIALS')
storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': storage_bucket
})

db = firestore.client()
bucket = storage.bucket()
