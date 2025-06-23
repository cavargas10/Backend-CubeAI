from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, Any, Optional
from services import user_service
from middleware.auth_middleware_fastapi import get_current_user

router = APIRouter(
    prefix="/user",
    tags=["User"]
)

@router.post("/register")
async def register_user(
    payload: Dict[str, Any], 
    user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        user_data = {
            "uid": user["uid"],
            "email": user["email"],
            "name": payload.get("name"),
            "profile_picture": payload.get("profile_picture", "")
        }
        user_service.register_user(user_data)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@router.get("/data")
async def get_user_data(user: Dict[str, Any] = Depends(get_current_user)):
    try:
        user_data = user_service.get_user_data(user["uid"])
        if user_data:
            return user_data
        else:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@router.post("/update/name")
async def update_name(
    payload: Dict[str, str], 
    user: Dict[str, Any] = Depends(get_current_user)
):
    new_name = payload.get("name")
    if not new_name or new_name.strip() == "":
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")
    
    try:
        updated_user_data = user_service.update_user_name(user["uid"], new_name)
        return updated_user_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@router.post("/update/profile-picture")
async def update_profile_picture(
    profile_picture: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    if not profile_picture:
        raise HTTPException(status_code=400, detail="No se proporcionó ninguna imagen")
    if profile_picture.filename == '':
        raise HTTPException(status_code=400, detail="No se seleccionó ningún archivo")

    try:
        profile_picture_url = user_service.update_profile_picture(user["uid"], profile_picture.file)
        return {"profile_picture": profile_picture_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@router.delete("/delete")
async def delete_user(user: Dict[str, Any] = Depends(get_current_user)):
    try:
        user_service.delete_user(user["uid"])
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")