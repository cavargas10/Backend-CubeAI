from flask import Blueprint, jsonify, request, current_app
from middleware.auth_middleware import verify_token_middleware
from services.img3d_service import img3d_service
from services.text3d_service import text3d_service
from services.textimg3d_service import textimg3d_service
from services.unico3d_service import unico3d_service
from services.multiimg3d_service import multiimg3d_service
from services.boceto3d_service import boceto3d_service

# Import existing and new Celery tasks
from tasks import run_text3d_generation
from tasks import run_img3d_generation, run_textimg3d_generation, run_unico3d_generation, run_multiimg3d_generation, run_boceto3d_generation

import os
import tempfile
import werkzeug.utils

bp = Blueprint('generation', __name__)

def save_temp_file(file_storage):
    if not file_storage:
        return None
    filename = werkzeug.utils.secure_filename(file_storage.filename)
    if not filename:
        filename = "uploaded_file"

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, filename)
    file_storage.save(temp_file_path)
    return temp_file_path

SERVICE_MAP = {
    "Imagen3D": img3d_service,
    "Texto3D": text3d_service,
    "TextImg3D": textimg3d_service,
    "Unico3D": unico3d_service,
    "MultiImagen3D": multiimg3d_service,
    "Boceto3D": boceto3d_service,
}

READABLE_TO_API_TYPE_MAP = {
    "Imagen a 3D": "Imagen3D",
    "Texto a 3D": "Texto3D",
    "Texto a Imagen a 3D": "TextImg3D",
    "Unico a 3D": "Unico3D",
    "Multi Imagen a 3D": "MultiImagen3D",
    "Boceto a 3D": "Boceto3D",
}

@bp.route("/imagen3D", methods=["POST"])
@verify_token_middleware
def predict_generation():
    try:
        user_uid = request.user["uid"]
        image_file = request.files.get("image")
        generation_name = request.form.get("generationName")

        if not image_file:
            return jsonify({"error": "Falta el archivo de imagen."}), 400
        if not generation_name:
            return jsonify({"error": "Falta el nombre de la generación."}), 400

        temp_file_path = save_temp_file(image_file)
        if not temp_file_path:
             return jsonify({"error": "No se pudo guardar el archivo temporal."}), 500

        task = run_img3d_generation.delay(user_uid, temp_file_path, generation_name)

        return jsonify({
            "message": "La generación de Imagen a 3D ha comenzado.",
            "task_id": task.id
        }), 202

    except ValueError as ve:
        current_app.logger.warning(f"Error de valor en /imagen3D: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error inesperado en /imagen3D: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

# VERSIÓN NUEVA (ASÍNCRONA)
@bp.route("/texto3D", methods=["POST"])
@verify_token_middleware
def create_text3d():
    try:
        user_uid = request.user["uid"]
        generation_name = request.json.get("generationName")
        user_prompt = request.json.get("prompt")
        selected_style = request.json.get("selectedStyle")

        if not all([generation_name, user_prompt, selected_style]):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        # ----> INICIO DEL CAMBIO <----

        # 1. En lugar de ejecutar la función directamente, la llamamos con .delay()
        # .delay() le pasa la tarea a Celery y devuelve el control inmediatamente.
        task = run_text3d_generation.delay(
            user_uid, generation_name, user_prompt, selected_style
        )
        
        # 2. Respondemos al frontend con el ID de la tarea.
        #    El frontend usará este ID para preguntar más tarde si la tarea ha terminado.
        #    El código 202 'Accepted' es el estándar para este tipo de respuesta.
        return jsonify({
            "message": "La generación ha comenzado. Te notificaremos cuando esté lista.",
            "task_id": task.id
        }), 202

        # ----> FIN DEL CAMBIO <----

    except ValueError as ve:
        # Los errores de validación (como nombre duplicado) se deben manejar aquí,
        # porque ocurren ANTES de que la tarea sea enviada.
        # Los errores de validación (como nombre duplicado) se deben manejar aquí si ocurren ANTES de que la tarea sea enviada.
        # Si _generation_exists es llamado dentro del servicio (y por ende, la tarea), este bloque no lo capturará.
        current_app.logger.warning(f"Error de valor en /texto3D: {ve}") # Catches pre-task submission errors
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error inesperado en /texto3D: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/textimg3D", methods=["POST"])
@verify_token_middleware
def create_textimg3d():
    try:
        user_uid = request.user["uid"]
        generation_name = request.json.get("generationName")
        subject = request.json.get("subject")
        style = request.json.get("style")
        additional_details = request.json.get("additionalDetails")

        if not all([generation_name, subject, style, additional_details]):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        task = run_textimg3d_generation.delay(user_uid, generation_name, subject, style, additional_details)

        return jsonify({
            "message": "La generación de Texto+Imagen a 3D ha comenzado.",
            "task_id": task.id
        }), 202
    except ValueError as ve:
        current_app.logger.warning(f"Error de valor en /textimg3D: {ve}") # Catches pre-task submission errors
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error inesperado en /textimg3D: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/unico3D", methods=["POST"])
@verify_token_middleware
def predict_unico3d():
    try:
        user_uid = request.user["uid"]
        image_file = request.files.get("image")
        generation_name = request.form.get("generationName")

        if not image_file:
            return jsonify({"error": "Falta el archivo de imagen."}), 400
        if not generation_name:
            return jsonify({"error": "Falta el nombre de la generación."}), 400

        temp_file_path = save_temp_file(image_file)
        if not temp_file_path:
             return jsonify({"error": "No se pudo guardar el archivo temporal."}), 500

        task = run_unico3d_generation.delay(user_uid, temp_file_path, generation_name)

        return jsonify({
            "message": "La generación de Unico3D ha comenzado.",
            "task_id": task.id
        }), 202
    except ValueError as ve:
        current_app.logger.warning(f"Error de valor en /unico3D: {ve}") # Catches pre-task submission errors
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error inesperado en /unico3D: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/multiimagen3D", methods=["POST"])
@verify_token_middleware
def predict_multi_image_3d():
    try:
        user_uid = request.user["uid"]
        frontal_image_file = request.files.get("frontal")
        lateral_image_file = request.files.get("lateral")
        trasera_image_file = request.files.get("trasera")
        generation_name = request.form.get("generationName")

        if not frontal_image_file or not lateral_image_file or not trasera_image_file:
            return jsonify({"error": "Por favor, cargue las tres imágenes (frontal, lateral y trasera)."}), 400
        if not generation_name:
            return jsonify({"error": "Por favor, ingrese un nombre para la generación."}), 400

        temp_frontal_path = save_temp_file(frontal_image_file)
        temp_lateral_path = save_temp_file(lateral_image_file)
        temp_trasera_path = save_temp_file(trasera_image_file)

        if not all([temp_frontal_path, temp_lateral_path, temp_trasera_path]):
            for p in [temp_frontal_path, temp_lateral_path, temp_trasera_path]:
                if p and os.path.exists(p):
                    os.remove(p)
                    try: os.rmdir(os.path.dirname(p))
                    except OSError: pass
            return jsonify({"error": "No se pudieron guardar todos los archivos temporales."}), 500

        task = run_multiimg3d_generation.delay(
            user_uid, temp_frontal_path, temp_lateral_path, temp_trasera_path, generation_name
        )

        return jsonify({
            "message": "La generación de Multi-Imagen a 3D ha comenzado.",
            "task_id": task.id
        }), 202
    except ValueError as ve:
        current_app.logger.warning(f"Error de valor en /multiimagen3D: {ve}") # Catches pre-task submission errors
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error inesperado en /multiimagen3D: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/boceto3D", methods=["POST"])
@verify_token_middleware
def predict_boceto_3d():
    try:
        user_uid = request.user["uid"]
        image_file = request.files.get("image")
        generation_name = request.form.get("generationName")
        description = request.form.get("description", "")

        if not image_file:
            return jsonify({"error": "Por favor, cargue una imagen del boceto."}), 400
        if not generation_name:
            return jsonify({"error": "Por favor, ingrese un nombre para la generación."}), 400

        temp_file_path = save_temp_file(image_file)
        if not temp_file_path:
             return jsonify({"error": "No se pudo guardar el archivo temporal."}), 500

        task = run_boceto3d_generation.delay(user_uid, temp_file_path, generation_name, description)

        return jsonify({
            "message": "La generación de Boceto a 3D ha comenzado.",
            "task_id": task.id
        }), 202
    except ValueError as ve:
        current_app.logger.warning(f"Error de valor en /boceto3D: {ve}") # Catches pre-task submission errors
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error inesperado en /boceto3D: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/generation/preview", methods=["POST"])
@verify_token_middleware
def upload_generation_preview():
    try:
        user_uid = request.user["uid"]
        preview_file = request.files.get("preview")
        generation_name = request.form.get("generation_name")
        prediction_type_api = request.form.get("prediction_type_api")

        if not all([preview_file, generation_name, prediction_type_api]):
            return jsonify({"error": "Faltan datos en la solicitud"}), 400

        if prediction_type_api in SERVICE_MAP:
            service_instance = SERVICE_MAP[prediction_type_api]
            updated_doc = service_instance.add_preview_image(user_uid, generation_name, preview_file)
            return jsonify(updated_doc), 200
        else:
            return jsonify({"error": "Tipo de predicción no válido"}), 400
    except Exception as e:
        current_app.logger.error(f"Error al subir la previsualización: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor al subir la previsualización"}), 500

@bp.route("/generations", methods=["GET"])
@verify_token_middleware
def get_user_generations():
    try:
        user_uid = request.user["uid"]
        generation_type_api = request.args.get('type')

        if generation_type_api in SERVICE_MAP:
            service_instance = SERVICE_MAP[generation_type_api]
            generations = service_instance.get_generations(user_uid)
            return jsonify(generations), 200
        else:
            return jsonify({"error": "Tipo de generación no válido"}), 400
    except Exception as e:
        current_app.logger.error(f"Error al obtener generaciones: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/generation", methods=["DELETE"])
@verify_token_middleware
def delete_generic_generation():
    try:
        user_uid = request.user["uid"]
        data = request.get_json()

        generation_name = data.get("generation_name")
        prediction_type_readable = data.get("prediction_type")

        if not generation_name or not prediction_type_readable:
            return jsonify({"error": "Faltan datos en la solicitud"}), 400

        generation_type_api = READABLE_TO_API_TYPE_MAP.get(prediction_type_readable)

        if generation_type_api and generation_type_api in SERVICE_MAP:
            service_instance = SERVICE_MAP[generation_type_api]
            success = service_instance.delete_generation(user_uid, generation_name)
            if success:
                return jsonify({"success": True}), 200
            else:
                return jsonify({"error": "Generación no encontrada"}), 404
        else:
            return jsonify({"error": f"Tipo de generación no válido: {prediction_type_readable}"}), 400
    except Exception as e:
        current_app.logger.error(f"Error al eliminar generación: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500