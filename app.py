import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from routes import generation_routes, user_routes
from queue_manager import queue_worker
from config.global_init import initialize_hf_token  

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando la aplicación...")
    
    initialize_hf_token()
    
    print("Creando el worker de la cola en segundo plano.")
    queue_task = asyncio.create_task(queue_worker())
    
    yield 
    
    print("Apagando la aplicación... Cancelando el worker.")
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        print("El worker de la cola ha sido cancelado exitosamente.")

app = FastAPI(
    lifespan=lifespan,
    title="CubeAI Backend",
    description="API para la generación de objetos 3D y gestión de usuarios.",
    version="1.0.0"
)

origins = [
    os.environ.get("FRONTEND_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generation_routes.router)
app.include_router(user_routes.router)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de CubeAI v2 con FastAPI"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)