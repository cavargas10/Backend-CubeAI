from fastapi import APIRouter, Depends, HTTPException, Body, Form, UploadFile, File
from typing import Dict, Any, Optional
from queue_manager import task_queue, create_job, jobs
from middleware.auth_middleware_fastapi import get_current_user

router = APIRouter(
    prefix="/generation",  
    tags=["Generation"]      
)

@router.post("/texto3D")
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
        job_id = create_job(job_type='texto3D', user_id=user_uid, data=job_data)

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

@router.post("/imagen3D")
async def enqueue_image3d_generation(
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    job_data = {
        "generation_name": generationName,
        "image_file": image  
    }

    try:
        job_id = create_job(job_type='imagen3D', user_id=user_uid, data=job_data)
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

@router.post("/textimg3D")
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
        job_id = create_job(job_type='textimg3D', user_id=user_uid, data=job_data)
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

@router.post("/unico3D")
async def enqueue_unico3d_generation(
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    job_data = {
        "generation_name": generationName,
        "image_file": image
    }

    try:
        job_id = create_job(job_type='unico3D', user_id=user_uid, data=job_data)
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

@router.post("/multiimagen3D")
async def enqueue_multi_image_3d_generation(
    generationName: str = Form(...),
    frontal: UploadFile = File(...),
    lateral: UploadFile = File(...),
    trasera: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    job_data = {
        "generation_name": generationName,
        "frontal_image": frontal,
        "lateral_image": lateral,
        "trasera_image": trasera
    }

    try:
        job_id = create_job(job_type='multiimagen3D', user_id=user_uid, data=job_data)
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

@router.post("/boceto3D")
async def enqueue_boceto_3d_generation(
    description: Optional[str] = Form(""),
    generationName: str = Form(...),
    image: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_uid = user["uid"]

    job_data = {
        "generation_name": generationName,
        "image_file": image,
        "description": description
    }

    try:
        job_id = create_job(job_type='boceto3D', user_id=user_uid, data=job_data)
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

@router.get("/status/{job_id}")
async def get_generation_status(job_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    if job["user_id"] != user["uid"]:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este trabajo")

    response = {"status": job["status"]}
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]
    
    if job["status"] == "pending":
        try:
            queue_items = list(task_queue._queue)
            position = queue_items.index(job_id) + 1
            response["position_in_queue"] = position
            response["queue_size"] = len(queue_items)
        except ValueError:
            pass
            
    return response