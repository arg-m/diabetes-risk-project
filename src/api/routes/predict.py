from fastapi import APIRouter, Depends

from ..schemas import DiabetesInput, DiabetesResponse
from ..dependencies import get_model_bundle
from ..services.predictor import predict_risk

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)

@router.post("/predict", response_model=DiabetesResponse)
def predict(
    data: DiabetesInput,
    bundle=Depends(get_model_bundle)
):
    return predict_risk(data, bundle)