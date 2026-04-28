import joblib
from contextlib import asynccontextmanager

from ..config import BASE_DIR, MODEL_FILENAME

model_bundle = None

@asynccontextmanager
async def lifespan(app):
    global model_bundle

    path = BASE_DIR / "models" / MODEL_FILENAME
    model_bundle = joblib.load(path)

    yield

def get_model_bundle():
    return model_bundle