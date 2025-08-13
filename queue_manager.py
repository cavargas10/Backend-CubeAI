import asyncio
import uuid
from typing import Dict, Any, Coroutine
from services import (
    text3d_service, img3d_service, textimg3d_service, 
    unico3d_service, multiimg3d_service, boceto3d_service,
    retexturize3d_service
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

task_queue = asyncio.Queue()

jobs: Dict[str, Dict[str, Any]] = {}

SERVICE_MAP: Dict[str, Coroutine] = {
    'Texto3D': text3d_service.create_text3d,
    'Imagen3D': img3d_service.create_generation,
    'TextoImagen2D': textimg3d_service.create_2d_image,
    'TextImg3D': textimg3d_service.create_3d_from_image, 
    'Unico3D': unico3d_service.create_unico3d,
    'MultiImagen3D': multiimg3d_service.create_multiimg3d,
    'Boceto3D': boceto3d_service.create_boceto3d,
    'Retexturize3D': retexturize3d_service.create_retexture3d,
}

SEMAPHORES: Dict[str, asyncio.Semaphore] = {
    'Texto3D': asyncio.Semaphore(10),
    'Imagen3D': asyncio.Semaphore(10),
    'TextoImagen2D': asyncio.Semaphore(10),
    'TextImg3D': asyncio.Semaphore(10),
    'Unico3D': asyncio.Semaphore(10),
    'MultiImagen3D': asyncio.Semaphore(10),
    'Boceto3D': asyncio.Semaphore(10),
    'Retexturize3D': asyncio.Semaphore(10),
}

def create_job(job_type: str, user_id: str, data: Dict[str, Any]) -> str:
    if job_type not in SEMAPHORES:
        logging.error(f"Intento de crear un trabajo para un tipo no configurado en SEMAPHORES: {job_type}")
        raise ValueError(f"El tipo de trabajo '{job_type}' no tiene un semáforo configurado.")
        
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "job_type": job_type,
        "user_id": user_id,
        "data": data,
        "result": None,
        "error": None
    }
    logging.info(f"Nuevo trabajo '{job_type}' creado con ID: {job_id} para el usuario {user_id}")
    return job_id

async def worker(worker_id: int):
    logging.info(f"Worker-{worker_id} ha iniciado.")
    while True:
        job_id = await task_queue.get()
        job_info = jobs.get(job_id)

        if not job_info:
            logging.warning(f"Worker-{worker_id} tomó un job_id ({job_id}) no válido. Saltando.")
            task_queue.task_done()
            continue
            
        job_type = job_info['job_type']
        semaphore = SEMAPHORES.get(job_type)

        logging.info(f"Worker-{worker_id} ha tomado el trabajo {job_id} ({job_type}). Esperando semáforo...")
        
        async with semaphore:
            logging.info(f"Worker-{worker_id} ha adquirido el semáforo para {job_type}. Procesando trabajo {job_id}.")
            jobs[job_id]["status"] = "processing"
            
            try:
                service_function = SERVICE_MAP.get(job_type)
                if not service_function:
                    raise ValueError(f"Tipo de trabajo desconocido: {job_type}")

                service_args = {
                    "user_uid": job_info['user_id'],
                    **job_info['data'] 
                }
                
                result_data = await service_function(**service_args)
                
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["result"] = result_data
                logging.info(f"Worker-{worker_id} completó exitosamente el trabajo {job_id}.")

            except Exception as e:
                logging.error(f"Worker-{worker_id} encontró un error procesando el trabajo {job_id}: {e}", exc_info=True)
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
            
            finally:
                logging.info(f"Worker-{worker_id} ha liberado el semáforo para {job_type}.")
                task_queue.task_done()