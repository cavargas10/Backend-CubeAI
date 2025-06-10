from config.firebase_config import bucket
import os

def upload_to_storage(file_source, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    
    if isinstance(file_source, str):
        if os.path.exists(file_source):
            blob.upload_from_filename(file_source)
        else:
            raise FileNotFoundError(f"El archivo local no se encontró en: {file_source}")
    else:
        file_source.seek(0)
        blob.upload_from_file(file_source)

    blob.make_public()
    return blob.public_url