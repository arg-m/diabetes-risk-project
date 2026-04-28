from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes.predict import router as predict_router
from .routes.health import router as health_router
from .dependencies import lifespan

from ..config import BASE_DIR

app = FastAPI(
    title="Diabetes Risk API",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)

frontend_dir = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse(frontend_dir / "index.html")