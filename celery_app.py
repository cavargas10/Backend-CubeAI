# Backend-CubeAI/celery_app.py

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery = Celery(
    'Backend-CubeAI',
    broker=os.getenv('REDIS_URL'),
    backend=os.getenv('REDIS_URL')
)

# ----> AÑADE ESTA LÍNEA <----
# Le dice a Celery que busque tareas automáticamente en el módulo 'tasks'
celery.autodiscover_tasks(['tasks']) 

celery.conf.update(
    task_track_started=True,
)