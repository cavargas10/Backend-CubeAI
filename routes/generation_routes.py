from flask import Blueprint, jsonify, request
from middleware.auth_middleware import verify_token_middleware
from services import generation_service, text3d_service, textimg3d_service, unico3d_service

bp = Blueprint('generation', __name__)

@bp.route("/imagen3D", methods=["POST"])
@verify_token_middleware
def predict_generation():
    try:
        image_file = request.files.get("image")
        generation_name = request.form.get("generationName")
        user_uid = request.user["uid"]
        
        prediction_result = generation_service.create_generation(user_uid, image_file, generation_name)
        return jsonify(prediction_result)
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

        result = text3d_service.create_text3d(user_uid, generation_name, user_prompt, selected_style)
        return jsonify(result)
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

        result = textimg3d_service.create_textimg3d(user_uid, generation_name, subject, style, additional_details)
        return jsonify(result)
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
        
        prediction_result = unico3d_service.create_unico3d(user_uid, image_file, generation_name)
        return jsonify(prediction_result)
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return jsonify({"error": str(ve)}), 400
    except KeyError as ke:
        print(f"Clave faltante: {ke}")
        return jsonify({"error": f"Clave faltante: {str(ke)}"}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500
    
@bp.route("/generations", methods=["GET"])
@verify_token_middleware
def get_user_generations():
    try:
        user_uid = request.user["uid"]
        generations = generation_service.get_user_generations(user_uid)
        text3d_generations = text3d_service.get_user_text3d_generations(user_uid)
        textimg3d_generations = textimg3d_service.get_user_textimg3d_generations(user_uid)
        unico3d_generations = unico3d_service.get_user_unico3d_generations(user_uid)
        return jsonify({
            "imagen3D": generations,
            "texto3D": text3d_generations,
            "textimg3D": textimg3d_generations,
            "unico3D": unico3d_generations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/generation/<generation_type>/<generation_name>", methods=["DELETE"])
@verify_token_middleware
def delete_generation(generation_type, generation_name):
    try:
        user_uid = request.user["uid"]
        
        if generation_type == "Imagen3D":
            success = generation_service.delete_generation(user_uid, generation_name)
        elif generation_type == "Texto3D":
            success = text3d_service.delete_text3d_generation(user_uid, generation_name)
        elif generation_type == "TextImg3D":
            success = textimg3d_service.delete_textimg3d_generation(user_uid, generation_name)
        elif generation_type == "Unico3D":
            success = unico3d_service.delete_unico3d_generation(user_uid, generation_name)
        else:
            return jsonify({"error": "Tipo de generación no válido"}), 400
        
        if success:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"error": "Generación no encontrada"}), 404
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": str(e)}), 500
