import asyncio
import uuid
from typing import Dict, Any
from services import (
    text3d_service, img3d_service, textimg3d_service, 
    unico3d_service, multiimg3d_service, boceto3d_service
)

task_queue = asyncio.Queue()
jobs: Dict[str, Dict[str, Any]] = {}

worker_is_processing = asyncio.Event()

SERVICE_MAP = {
    'Texto3D': text3d_service.create_text3d,
    'Imagen3D': img3d_service.create_generation,
    'TextoImagen2D': textimg3d_service.create_2d_image,
    'TextImg3D': textimg3d_service.create_3d_from_image, 
    'Unico3D': unico3d_service.create_unico3d,
    'MultiImagen3D': multiimg3d_service.create_multiimg3d,
    'Boceto3D': boceto3d_service.create_boceto3d,
}

def create_job(job_type: str, user_id: str, data: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "job_type": job_type,
        "user_id": user_id,
        "data": data,
        "result": None,
        "error": None
    }
    print(f"Nuevo trabajo '{job_type}' creado con ID: {job_id}")
    return job_id


async def queue_worker():
    print("El worker de la cola ha iniciado...")
    while True:
        job_id = await task_queue.get()
        
        job_info = jobs.get(job_id)
        if not job_info:
            print(f"ADVERTENCIA: Se recibió un job_id ({job_id}) no encontrado. Saltando.")
            task_queue.task_done()
            continue

        print(f"Worker ha tomado el trabajo: {job_id} (Tipo: {job_info['job_type']})")

        worker_is_processing.set() 
        jobs[job_id]["status"] = "processing"
        
        try:
            service_function = SERVICE_MAP.get(job_info['job_type'])
            
            if not service_function:
                raise ValueError(f"Tipo de trabajo desconocido: {job_info['job_type']}")

            service_args = {
                "user_uid": job_info['user_id'],
                **job_info['data'] 
            }
            
            result_data = await service_function(**service_args)
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result_data
            print(f"Trabajo {job_id} completado.")

        except Exception as e:
            print(f"Error procesando el trabajo {job_id}: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        finally:
            worker_is_processing.clear()
            task_queue.task_done()