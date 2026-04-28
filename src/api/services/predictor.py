import numpy as np
import pandas as pd

from fastapi import HTTPException

from ...config import FEATURE_LABELS, SELECTED_FEATURES
from ..schemas import DiabetesResponse

def get_risk_level(prob):
    if prob < 0.30:
        return "Низкий риск"
    elif prob < 0.70:
        return "Средний риск"
    return "Высокий риск"

def predict_risk(data, bundle):
    if bundle is None:
        raise HTTPException(503, "Модель не загружена")
    
    df = pd.DataFrame([data.model_dump()])[SELECTED_FEATURES]

    model = bundle["model"]
    prep = bundle["preprocessor"]

    X = prep.transform(df)
    prob = float(model.predict_proba(X)[0][1])

    if bundle["model_name"] == "logistic":
        imp = np.abs(model.coef_[0])
    else:
        imp = model.feature_importances_

    top_idx = np.argsort(imp)[::-1][:3]
    top = [
        FEATURE_LABELS[SELECTED_FEATURES[i]]
        for i in top_idx
    ]

    level = get_risk_level(prob)

    return DiabetesResponse(
        probability=round(prob, 3),
        risk_level=level,
        top_factors=top,
        message=f"Риск диабета: {level} ({prob:.1%})"
    )