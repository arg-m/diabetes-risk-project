"""
FastAPI Backend для предсказания риска диабета.
Загружает обученный Pipeline + модель, валидирует входные данные, возвращает интерпретируемый результат.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from .schemas import DiabetesInput, DiabetesResponse
from ..config import BASE_DIR, MODEL_FILENAME, FEATURE_LABELS, SELECTED_FEATURES

model_package = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте приложения и корректное освобождение ресурсов."""
    global model_package
    model_path = BASE_DIR / "models" / MODEL_FILENAME
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}\nСначала выполните обучение (Шаг 3).")
        
    model_package = joblib.load(model_path)
    print(f"Модель успешно загружена: {model_package['model_name']} | Версия: {model_package['metrics']['roc_auc']:.3f} ROC-AUC")
    
    yield
    
    print("Приложение завершает работу. Модель выгружена из памяти.")

app = FastAPI(
    title="Diabetes Risk API",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan
)

# Разрешаем CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_risk_level(prob: float) -> str:
    """Классификация уровня риска по вероятности."""
    if prob < 0.3:
        return "Низкий риск"
    elif prob < 0.7:
        return "Средний риск"
    return "Высокий риск"

@app.post("/api/v1/predict", response_model=DiabetesResponse)
async def predict(data: DiabetesInput):
    """
    Принимает 7 признаков, применяет предобработку, возвращает вероятность, уровень риска и топ-факторы.
    """
    if model_package is None:
        raise HTTPException(status_code=503, detail="Модель не инициализирована")

    try:
        # 1. Формируем DataFrame в строгом порядке признаков из config
        input_df = pd.DataFrame([data.model_dump()])[SELECTED_FEATURES]
        
        # 2. Применяем сохранённый пайплайн и предсказываем вероятность диабета (класс 1)
        pipeline = model_package["pipeline"]
        model = model_package["model"]
        proba = model.predict_proba(pipeline.transform(input_df))[0][1]
        prob = float(proba)
        
        # 3. Определяем топ-3 фактора влияния (абсолютные коэффициенты/важности)
        if model_package["model_name"] == "logistic":
            importances = np.abs(model.coef_[0])
        else:
            importances = model.feature_importances_
            
        top_indices = np.argsort(importances)[::-1][:3]
        top_factors = [FEATURE_LABELS.get(SELECTED_FEATURES[i], SELECTED_FEATURES[i]) for i in top_indices]
        
        # 4. Формируем ответ
        risk_level = get_risk_level(prob)
        
        return DiabetesResponse(
            probability=round(prob, 3),
            risk_level=risk_level,
            top_factors=top_factors,
            message=f"Риск развития диабета: {risk_level} ({prob:.1%})"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки запроса: {str(e)}")

@app.get("/")
def health_check():
    """Проверка работоспособности API."""
    return {
        "status": "API is running",
        "model_loaded": model_package is not None,
        "docs": "/docs"
    }

frontend_dir = Path(__file__).parent.parent.parent / "frontend"

@app.get("/app")
async def serve_frontend():
    return FileResponse(frontend_dir / "index.html")

app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")