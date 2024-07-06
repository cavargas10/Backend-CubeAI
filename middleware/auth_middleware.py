from firebase_admin import auth
from flask import request, jsonify
from functools import wraps

def verify_token_middleware(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header missing or invalid"}), 403

        id_token = auth_header.split(" ")[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.user = decoded_token
        except Exception as e:
            return jsonify({"error": f"Token verification failed: {str(e)}"}), 403

        return f(*args, **kwargs)
    return decorator