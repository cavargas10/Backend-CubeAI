import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from routes import generation_routes, user_routes
from queue_manager import worker 
from config.global_init import initialize_hf_token
import logging

NUM_WORKERS = 20 

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Iniciando la aplicación...")
    
    initialize_hf_token()
    
    worker_tasks = []
    
    logging.info(f"Creando un pool de {NUM_WORKERS} workers en segundo plano.")
    for i in range(NUM_WORKERS):
        task = asyncio.create_task(worker(worker_id=i + 1))
        worker_tasks.append(task)
    
    yield 
    
    logging.info("Apagando la aplicación... Cancelando los workers.")
    for task in worker_tasks:
        task.cancel()
    
    try:
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        logging.info("Todos los workers han sido cancelados exitosamente.")
    except asyncio.CancelledError:
        logging.info("Los workers han sido cancelados durante el apagado.")

app = FastAPI(
    lifespan=lifespan,
    title="Instant3D Backend",
    description="API para la generación de objetos 3D y gestión de usuarios con pool de workers.",
    version="1.1.0"
)

origins = [
    os.environ.get("FRONTEND_URL", "http://localhost:5173"),
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
    return {"message": "Bienvenido a la API de CubeAI v2 con FastAPI y Pool de Workers"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)