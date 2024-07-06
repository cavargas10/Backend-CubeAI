from flask import Blueprint, jsonify, request, current_app
from middleware.auth_middleware import verify_token_middleware
from services import user_service

bp = Blueprint('user', __name__)

@bp.route("/register_user", methods=["POST"])
@verify_token_middleware
def register_user():
    try:
        user_data = {
            "uid": request.user["uid"],
            "email": request.user["email"],
            "name": request.json.get("name"),
            "profile_picture": request.json.get("profile_picture", "")
        }
        user_service.register_user(user_data)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/user_data", methods=["GET"])
@verify_token_middleware
def get_user_data():
    try:
        user_uid = request.user["uid"]
        user_data = user_service.get_user_data(user_uid)
        if user_data:
            return jsonify(user_data), 200
        else:
            return jsonify({"error": "Usuario no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/update_name", methods=["POST"])
@verify_token_middleware
def update_name():
    try:
        user_uid = request.user["uid"]
        new_name = request.json.get("name")
        if not new_name or new_name.strip() == "":
            return jsonify({"error": "El nombre no puede estar vacío"}), 400
        updated_user_data = user_service.update_user_name(user_uid, new_name)
        return jsonify(updated_user_data), 200
    except Exception as e:
        current_app.logger.error(f"Error en update_name: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/update_profile_picture", methods=["POST"])
@verify_token_middleware
def update_profile_picture():
    try:
        user_uid = request.user["uid"]
        if 'profile_picture' not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen"}), 400
        file = request.files['profile_picture']
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        if file:
            profile_picture_url = user_service.update_profile_picture(user_uid, file)
            return jsonify({"profile_picture": profile_picture_url}), 200
    except Exception as e:
        current_app.logger.error(f"Error en update_profile_picture: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@bp.route("/delete_user", methods=["DELETE"])
@verify_token_middleware
def delete_user():
    try:
        user_uid = request.user["uid"]
        user_service.delete_user(user_uid)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500