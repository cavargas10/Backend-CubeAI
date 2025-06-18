from flask import Flask, jsonify
from flask_cors import CORS
from routes import user_routes, generation_routes

app = Flask(__name__)
CORS(app)

# New root route
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the API"}), 200

app.register_blueprint(user_routes.bp)
app.register_blueprint(generation_routes.bp)

if __name__ == "__main__":
    app.run(debug=True, port=8080)