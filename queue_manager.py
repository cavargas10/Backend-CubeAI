import asyncio
import uuid
from typing import Dict, Any
from services.text3d_service import text3d_service
from services.img3d_service import img3d_service
from services.textimg3d_service import textimg3d_service
from services.unico3d_service import unico3d_service
from services.multiimg3d_service import multiimg3d_service
from services.boceto3d_service import boceto3d_service

task_queue = asyncio.Queue()
jobs: Dict[str, Dict[str, Any]] = {}

SERVICE_MAP = {
    'texto3D': text3d_service.create_text3d,
    'imagen3D': img3d_service.create_generation,
    'textimg3D': textimg3d_service.create_textimg3d,
    'unico3D': unico3d_service.create_unico3d,
    'multiimagen3D': multiimg3d_service.create_multiimg3d, 
    'boceto3D': boceto3d_service.create_boceto3d,      
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
            
        task_queue.task_done()