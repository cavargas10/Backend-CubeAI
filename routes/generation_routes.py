from fastapi import APIRouter, Depends, HTTPException, Body, Form, UploadFile, File
from typing import Dict, Any, Optional
from queue_manager import task_queue, create_job, jobs
from middleware.auth_middleware_fastapi import get_current_user
from services import SERVICE_INSTANCE_MAP
import logging
from config.firebase_config import db
import datetime

router = APIRouter(
    prefix="/generation",  
    tags=["Generation"]      
)

async def enqueue_job(job_type: str, user_uid: str, job_data: Dict[str, Any]):
    try:
        job_id = create_job(job_type=job_type, user_id=user_uid, data=job_data)
        await task_queue.put(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "El trabajo ha sido añadido a la cola de procesamiento."
        }
    except ValueError as ve:
        logging.warning(f"Conflicto al crear trabajo para el usuario {user_uid}: {str(ve)}")
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        logging.error(f"Error interno al encolar trabajo '{job_type}' para el usuario {user_uid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al encolar el trabajo: {e}")

@router.post("/Texto3D")
async def enqueue_text3d_generation(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
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
    
    return await enqueue_job('Texto3D', user["uid"], job_data)

@router.post("/Imagen3D")
async def enqueue_image3d_generation(
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    job_data = {
        "generation_name": generationName,
        "image_bytes": image_bytes, 
        "image_filename": image.filename 
    }

    return await enqueue_job('Imagen3D', user["uid"], job_data)

@router.post("/TextoImagen2D")
async def enqueue_text_to_2d_image_generation(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    generation_name = payload.get("generationName")
    prompt = payload.get("prompt")
    selected_style = payload.get("selectedStyle")

    if not all([generation_name, prompt]):
         raise HTTPException(status_code=400, detail="Faltan los campos 'generationName' y 'prompt'.")
    
    job_data = {
        "generation_name": generation_name,
        "prompt": prompt,
        "selected_style": selected_style or "none",
    }
    
    return await enqueue_job('TextoImagen2D', user["uid"], job_data)

@router.post("/TextImg3D")
async def enqueue_textimg3d_generation(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    generation_name = payload.get("generationName")
    image_url = payload.get("imageUrl")
    prompt = payload.get("prompt")
    selected_style = payload.get("selectedStyle")

    if not all([generation_name, image_url, prompt, selected_style]):
        raise HTTPException(status_code=400, detail="Faltan campos requeridos: generationName, imageUrl, prompt, y selectedStyle.")

    job_data = { 
        "generation_name": generation_name, 
        "image_url": image_url,
        "prompt": prompt,
        "selected_style": selected_style
    }
    
    return await enqueue_job('TextImg3D', user["uid"], job_data)

@router.post("/Unico3D")
async def enqueue_unico3d_generation(
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")
    
    job_data = {
        "generation_name": generationName,
        "image_bytes": image_bytes, 
        "image_filename": image.filename 
    }
    
    return await enqueue_job('Unico3D', user["uid"], job_data)

@router.post("/MultiImagen3D")
async def enqueue_multi_image_3d_generation(
    generationName: str = Form(...),
    frontal: UploadFile = File(...),
    lateral: UploadFile = File(...),
    trasera: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
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

    return await enqueue_job('MultiImagen3D', user["uid"], job_data)

@router.post("/Boceto3D")
async def enqueue_boceto_3d_generation(
    description: Optional[str] = Form(""),
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    job_data = {
        "generation_name": generationName,
        "image_bytes": image_bytes,
        "image_filename": image.filename,
        "description": description
    }
    
    return await enqueue_job('Boceto3D', user["uid"], job_data)
    
@router.post("/Retexturize3D")
async def enqueue_retexturize_3d_generation(
    generationName: str = Form(...),
    model: UploadFile = File(...),
    texture: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    model_bytes = await model.read()
    texture_bytes = await texture.read()

    if not model_bytes or not texture_bytes:
        raise HTTPException(status_code=400, detail="El archivo del modelo y de la textura no pueden estar vacíos.")

    job_data = {
        "generation_name": generationName,
        "model_bytes": model_bytes,
        "model_filename": model.filename,
        "texture_bytes": texture_bytes,
        "texture_filename": texture.filename
    }

    return await enqueue_job('Retexturize3D', user["uid"], job_data)

@router.put("/{prediction_type}/{generation_name}")
async def regenerate_generation(
    prediction_type: str,
    generation_name: str,
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    service_instance = SERVICE_INSTANCE_MAP.get(prediction_type)
    if not service_instance:
        raise HTTPException(status_code=400, detail=f"Tipo de predicción no válido: {prediction_type}")

    success = service_instance.clear_generation_storage(user_uid=user["uid"], generation_name=generation_name)
    if not success:
        raise HTTPException(status_code=500, detail="Error al limpiar la generación anterior. Inténtalo de nuevo.")

    doc_ref = db.collection('predictions').document(user["uid"]).collection(prediction_type).document(generation_name)
    doc_ref.update({
        "modelUrl": None,
        "previewImageUrl": None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

    job_data = payload
    job_data['generation_name'] = generation_name
    
    return await enqueue_job(prediction_type, user["uid"], job_data)

@router.get("/status/{job_id}")
async def get_generation_status(job_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    if job["user_id"] != user["uid"]:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este trabajo")

    status_to_report = job["status"]
    if status_to_report == "pending":
        status_to_report = "queued"

    response = {"status": status_to_report}
    
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]
        
    return response

@router.get("/history/{generation_type}")
async def get_user_generations(
    generation_type: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    service_instance = SERVICE_INSTANCE_MAP.get(generation_type)
    if service_instance:
        try:
            generations = service_instance.get_generations(user_uid=user["uid"])
            return generations
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener el historial: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de generación no válido: {generation_type}")

@router.post("/preview")
async def upload_generation_preview(
    preview: UploadFile = File(...),
    generation_name: str = Form(...),
    prediction_type_api: str = Form(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    service_instance = SERVICE_INSTANCE_MAP.get(prediction_type_api)
    if service_instance:
        try:
            updated_doc = service_instance.add_preview_image(
                user_uid=user["uid"], 
                generation_name=generation_name, 
                preview_file=preview.file
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
    service_instance = SERVICE_INSTANCE_MAP.get(prediction_type)
    if service_instance:
        try:
            success = service_instance.delete_generation(user_uid=user["uid"], generation_name=generation_name)
            if success:
                return {"success": True, "message": "Generación eliminada correctamente."}
            else:
                raise HTTPException(status_code=404, detail="Generación no encontrada para eliminar")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error interno al eliminar la generación: {e}")