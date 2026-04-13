"""
Pydantic-схемы для валидации входных и выходных данных API.
Совместимы с финальным списком 7 признаков из config.py.
"""
from pydantic import BaseModel, Field
from typing import List

class DiabetesInput(BaseModel):
    GenHlth: float = Field(..., ge=1, le=5, description="Общее состояние здоровья (1-5)")
    BMI: float = Field(..., gt=0, description="Индекс массы тела")
    Age: float = Field(..., ge=1, le=13, description="Возрастная группа (1-13)")
    HighBP: int = Field(..., ge=0, le=1, description="Высокое кровяное давление (0/1)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Физическая активность (0/1)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Болезни сердца/инфаркт (0/1)")
    DiffWalk: int = Field(..., ge=0, le=1, description="Трудности при ходьбе (0/1)")

class DiabetesResponse(BaseModel):
    probability: float = Field(..., description="Вероятность диабета (0.0 - 1.0)")
    risk_level: str = Field(..., description="Категория риска: Низкий/Средний/Высокий")
    top_factors: List[str] = Field(..., description="Топ-3 фактора, влияющих на риск")
    message: str = Field(..., description="Человекочитаемое пояснение")