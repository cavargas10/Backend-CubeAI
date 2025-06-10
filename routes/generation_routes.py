from flask import Blueprint, jsonify, request
from middleware.auth_middleware import verify_token_middleware
from services import img3d_service, text3d_service, textimg3d_service, unico3d_service, multiimg3d_service, boceto3d_service

bp = Blueprint('generation', __name__)

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
        image_file = request.files.get("image")
        generation_name = request.form.get("generationName")
        user_uid = request.user["uid"]
        
        prediction_img3d_result = img3d_service.create_generation(user_uid, image_file, generation_name)
        return jsonify(prediction_img3d_result)
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400
    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

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

        prediction_text3d_result = text3d_service.create_text3d(user_uid, generation_name, user_prompt, selected_style)
        return jsonify(prediction_text3d_result)
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400
    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

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

        prediction_textimg3d_result = textimg3d_service.create_textimg3d(user_uid, generation_name, subject, style, additional_details)
        return jsonify(prediction_textimg3d_result)
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400
    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500
    
@bp.route("/unico3D", methods=["POST"])
@verify_token_middleware
def predict_unico3d():
    try:
        user_uid = request.user["uid"]
        image_file = request.files.get("image")
        generation_name = request.form.get("generationName")
        
        if not image_file or not generation_name:
            return jsonify({"error": "Faltan campos requeridos: imagen y/o nombre de generación"}), 400
        
        prediction_unico3d_result = unico3d_service.create_unico3d(user_uid, image_file, generation_name)
        return jsonify(prediction_unico3d_result)
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400
    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@bp.route("/multiimagen3D", methods=["POST"])
@verify_token_middleware
def predict_multi_image_3d():
    try:
        frontal_image = request.files.get("frontal")
        lateral_image = request.files.get("lateral")
        trasera_image = request.files.get("trasera")
        generation_name = request.form.get("generationName")
        user_uid = request.user["uid"]  
        
        if not frontal_image or not lateral_image or not trasera_image:
            raise ValueError("Por favor, cargue las tres imágenes (frontal, lateral y trasera).")

        if not generation_name:
            raise ValueError("Por favor, ingrese un nombre para la generación.")

        prediction_multiimg3d_result = multiimg3d_service.create_multiimg3d(
            user_uid=user_uid,
            frontal_image=frontal_image,
            lateral_image=lateral_image,
            trasera_image=trasera_image,
            generation_name=generation_name
        )
        return jsonify(prediction_multiimg3d_result)
    
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400
    
    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400
    
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@bp.route("/boceto3D", methods=["POST"])
@verify_token_middleware
def predict_boceto_3d():
    try:
        image_file = request.files.get("image")  
        generation_name = request.form.get("generationName")  
        description = request.form.get("description", "")  
        user_uid = request.user["uid"]  

        if not image_file:
            raise ValueError("Por favor, cargue una imagen del boceto.")
        if not generation_name:
            raise ValueError("Por favor, ingrese un nombre para la generación.")
        prediction_boceto3d_result = boceto3d_service.create_boceto3d(
            user_uid=user_uid,
            image_file=image_file,
            generation_name=generation_name,
            description=description
        )

        return jsonify(prediction_boceto3d_result)

    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400

    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400

    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@bp.route("/generation/preview", methods=["POST"])
@verify_token_middleware
def upload_generation_preview():
    try:
        user_uid = request.user["uid"]
        preview_file = request.files.get("preview")
        generation_name = request.form.get("generation_name")
        prediction_type_api = request.form.get("prediction_type_api")

        if not all([preview_file, generation_name, prediction_type_api]):
            return jsonify({"error": "Faltan datos en la solicitud (preview, generation_name, prediction_type_api)"}), 400

        if prediction_type_api in SERVICE_MAP:
            service_module = SERVICE_MAP[prediction_type_api]
            updated_doc = service_module.add_preview_image(user_uid, generation_name, preview_file)
            return jsonify(updated_doc), 200
        else:
            return jsonify({"error": "Tipo de predicción no válido"}), 400
    except Exception as e:
        print(f"Error al subir la previsualización: {e}")
        return jsonify({"error": "Error interno del servidor al subir la previsualización"}), 500
    
@bp.route("/generations", methods=["GET"])
@verify_token_middleware
def get_user_generations():
    try:
        user_uid = request.user["uid"]
        generation_type_api = request.args.get('type') 

        if generation_type_api in SERVICE_MAP:
            service_module = SERVICE_MAP[generation_type_api]
            generations = service_module.get_generations(user_uid)
            return jsonify(generations), 200
        else:
            return jsonify({"error": "Tipo de generación no válido"}), 400
    except Exception as e:
        print(f"Error al obtener generaciones: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

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
            service_module = SERVICE_MAP[generation_type_api]
            success = service_module.delete_generation(user_uid, generation_name)
            if success:
                return jsonify({"success": True}), 200
            else:
                return jsonify({"error": "Generación no encontrada"}), 404
        else:
            return jsonify({"error": f"Tipo de generación no válido: {prediction_type_readable}"}), 400
            
    except Exception as e:
        print(f"Error al eliminar generación: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500