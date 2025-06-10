from flask import Blueprint, jsonify, request
from middleware.auth_middleware import verify_token_middleware
from services import img3d_service, text3d_service, textimg3d_service, unico3d_service, multiimg3d_service, boceto3d_service

bp = Blueprint('generation', __name__)

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
    
service_map = {
    "Imagen3D": img3d_service.get_user_generations,
    "Texto3D": text3d_service.get_user_text3d_generations,
    "TextImg3D": textimg3d_service.get_user_textimg3d_generations,
    "Unico3D": unico3d_service.get_user_unico3d_generations,
    "MultiImagen3D": multiimg3d_service.get_user_multiimg3d_generations,
    "Boceto3D": boceto3d_service.get_user_boceto3d_generations,
}

@bp.route("/generations", methods=["GET"])
@verify_token_middleware
def get_user_generations():
    try:
        user_uid = request.user["uid"]
        generation_type = request.args.get('type')

        if generation_type:
            if generation_type in service_map:
                get_generations_func = service_map[generation_type]
                generations = get_generations_func(user_uid)
                return jsonify(generations), 200
            else:
                return jsonify({"error": "Tipo de generación no válido"}), 400
        else:
            all_generations = {
                "imagen3D": service_map["Imagen3D"](user_uid),
                "texto3D": service_map["Texto3D"](user_uid),
                "textimg3D": service_map["TextImg3D"](user_uid),
                "unico3D": service_map["Unico3D"](user_uid),
                "multiimg3D": service_map["MultiImagen3D"](user_uid),
                "boceto3D": service_map["Boceto3D"](user_uid),
            }
            return jsonify(all_generations), 200

    except Exception as e:
        print(f"Error al obtener generaciones: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@bp.route("/generation/<generation_type>/<generation_name>", methods=["DELETE"])
@verify_token_middleware
def delete_generation(generation_type, generation_name):
    try:
        user_uid = request.user["uid"]
        
        if generation_type == "Imagen3D":
            success = img3d_service.delete_generation(user_uid, generation_name)
        elif generation_type == "Texto3D":
            success = text3d_service.delete_text3d_generation(user_uid, generation_name)
        elif generation_type == "TextImg3D":
            success = textimg3d_service.delete_textimg3d_generation(user_uid, generation_name)
        elif generation_type == "Unico3D":
            success = unico3d_service.delete_unico3d_generation(user_uid, generation_name)
        elif generation_type == "MultiImagen3D":
            success = multiimg3d_service.delete_multiimg3d_generation(user_uid, generation_name)
        elif generation_type == "Boceto3D":
            success = boceto3d_service.delete_boceto3d_generation(user_uid, generation_name)
        else:
            return jsonify({"error": "Tipo de generación no válido"}), 400
        
        if success:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"error": "Generación no encontrada"}), 404
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": str(e)}), 500