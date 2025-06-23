from fastapi import APIRouter, Depends, HTTPException, Body, Form, UploadFile, File
from typing import Dict, Any, Optional
from queue_manager import task_queue, create_job, jobs, worker_is_processing 
from middleware.auth_middleware_fastapi import get_current_user
from services import SERVICE_INSTANCE_MAP

router = APIRouter(
    prefix="/generation",  
    tags=["Generation"]      
)

@router.post("/Texto3D")
async def enqueue_text3d_generation(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    generation_name = payload.get("generationName")
    user_prompt = payload.get("prompt")
    selected_style = payload.get("selectedStyle")

    if not all([generation_name, user_prompt, selected_style]):
        raise HTTPException(status_code=400, detail="Faltan campos requeridos: generationName, prompt, selectedStyle")

    job_data = {
        "generation_name": generation_name,
        "user_prompt": user_prompt,
        "selected_style": selected_style
    }
    
    try:
        job_id = create_job(job_type='Texto3D', user_id=user_uid, data=job_data)

        await task_queue.put(job_id)

        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": task_queue.qsize()
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.post("/Imagen3D")
async def enqueue_image3d_generation(
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    job_data = {
        "generation_name": generationName,
        "image_bytes": image_bytes, 
        "image_filename": image.filename 
    }

    try:
        job_id = create_job(job_type='Imagen3D', user_id=user_uid, data=job_data)
        await task_queue.put(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": task_queue.qsize()
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.post("/TextImg3D")
async def enqueue_textimg3d_generation(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]
    
    generation_name = payload.get("generationName")
    subject = payload.get("subject")
    style = payload.get("style")
    additional_details = payload.get("additionalDetails")

    if not all([generation_name, subject, style, additional_details]):
        raise HTTPException(status_code=400, detail="Faltan campos requeridos")

    job_data = {
        "generation_name": generation_name,
        "subject": subject,
        "style": style,
        "additional_details": additional_details
    }
    
    try:
        job_id = create_job(job_type='TextImg3D', user_id=user_uid, data=job_data)
        await task_queue.put(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": task_queue.qsize()
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.post("/Unico3D")
async def enqueue_unico3d_generation(
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")
    
    job_data = {
        "generation_name": generationName,
        "image_bytes": image_bytes, 
        "image_filename": image.filename 
    }

    try:
        job_id = create_job(job_type='Unico3D', user_id=user_uid, data=job_data)
        await task_queue.put(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": task_queue.qsize()
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.post("/MultiImagen3D")
async def enqueue_multi_image_3d_generation(
    generationName: str = Form(...),
    frontal: UploadFile = File(...),
    lateral: UploadFile = File(...),
    trasera: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    frontal_bytes = await frontal.read()
    lateral_bytes = await lateral.read()
    trasera_bytes = await trasera.read()

    if not all([frontal_bytes, lateral_bytes, trasera_bytes]):
        raise HTTPException(status_code=400, detail="Uno o más archivos de imagen están vacíos.")

    job_data = {
        "generation_name": generationName,
        "frontal_bytes": frontal_bytes,
        "lateral_bytes": lateral_bytes,
        "trasera_bytes": trasera_bytes,
        "filenames": {
            "frontal": frontal.filename,
            "lateral": lateral.filename,
            "trasera": trasera.filename
        }
    }

    try:
        job_id = create_job(job_type='MultiImagen3D', user_id=user_uid, data=job_data)
        await task_queue.put(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": task_queue.qsize()
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.post("/Boceto3D")
async def enqueue_boceto_3d_generation(
    description: Optional[str] = Form(""),
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    job_data = {
        "generation_name": generationName,
        "image_bytes": image_bytes,
        "image_filename": image.filename,
        "description": description
    }

    try:
        job_id = create_job(job_type='Boceto3D', user_id=user_uid, data=job_data)
        await task_queue.put(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": task_queue.qsize()
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.get("/history/{generation_type}")
async def get_user_generations(
    generation_type: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]
    
    service_instance = SERVICE_INSTANCE_MAP.get(generation_type)
    
    if service_instance:
        try:
            generations = service_instance.get_generations(user_uid)
            return generations
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener el historial: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de generación no válido: {generation_type}")

@router.get("/status/{job_id}")
async def get_generation_status(job_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    if job["user_id"] != user["uid"]:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este trabajo")

    response = {"status": job["status"]}

    if job["status"] == "pending":
        try:
            queue_items = list(task_queue._queue)
            position_in_queue = queue_items.index(job_id) + 1

            total_in_queue = len(queue_items)

            if worker_is_processing.is_set():
                response["status"] = "queued"
                response["position_in_queue"] = position_in_queue + 1
                response["queue_size"] = total_in_queue + 1
            else:
                response["status"] = "queued"
                response["position_in_queue"] = position_in_queue
                response["queue_size"] = total_in_queue
                
        except ValueError:
            pass

    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]
        
    return response

@router.post("/preview")
async def upload_generation_preview(
    preview: UploadFile = File(...),
    generation_name: str = Form(...),
    prediction_type_api: str = Form(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    if not all([preview, generation_name, prediction_type_api]):
        raise HTTPException(status_code=400, detail="Faltan datos en la solicitud")

    service_instance = SERVICE_INSTANCE_MAP.get(prediction_type_api)

    if service_instance:
        try:
            updated_doc = service_instance.add_preview_image(
                user_uid, 
                generation_name, 
                preview.file
            )
            return updated_doc
        except ValueError as ve:
            raise HTTPException(status_code=404, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error interno al subir la previsualización: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de predicción no válido: {prediction_type_api}")

@router.delete("/{prediction_type}/{generation_name}")
async def delete_specific_generation(
    prediction_type: str,
    generation_name: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    service_instance = SERVICE_INSTANCE_MAP.get(prediction_type)
    
    if service_instance:
        try:
            success = service_instance.delete_generation(user_uid, generation_name)
            if success:
                return {"success": True, "message": "Generación eliminada correctamente."}
            else:
                raise HTTPException(status_code=404, detail="Generación no encontrada")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error interno al eliminar la generación: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de predicción no válido: {prediction_type}")