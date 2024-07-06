from flask import Flask
from flask_cors import CORS
from routes import user_routes, generation_routes

app = Flask(__name__)
CORS(app)

app.register_blueprint(user_routes.bp)
app.register_blueprint(generation_routes.bp)

if __name__ == "__main__":
    app.run(debug=True, port=8080)