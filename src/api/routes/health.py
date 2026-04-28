from fastapi import APIRouter
from ..dependencies import get_model_bundle

router = APIRouter(tags=["Health"])

@router.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": get_model_bundle() is not None,
        "docs": "/docs"
    }

@router.get("/health")
def health():
    return {"status": "healthy"}