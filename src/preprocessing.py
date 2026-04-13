"""
Модуль предобработки данных: загрузка, сплит, кодирование, масштабирование.
Возвращает готовый sklearn.Pipeline, совместимый с обучением и FastAPI.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pathlib import Path


try:
    from config import (
        DATA_RAW, DATASET_FILENAME, TARGET_COL, SELECTED_FEATURES,
        RANDOM_STATE, TEST_SIZE
    )
except ImportError:
    from src.config import (
        DATA_RAW, DATASET_FILENAME, TARGET_COL, SELECTED_FEATURES,
        RANDOM_STATE, TEST_SIZE
    )

def load_data():
    data_path = DATA_RAW / DATASET_FILENAME
    if not data_path.exists():
        raise FileNotFoundError(f"Файл не найден: {data_path}")
        
    df = pd.read_csv(data_path)
    required = SELECTED_FEATURES + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")
        
    return df[required].copy()

def build_preprocessing_pipeline():
    """
    Создаёт ColumnTransformer для разнотипных признаков.
    Гарантирует строгий порядок и воспроизводимость.
    """
    numeric_features = ['BMI']
    ordinal_features = ['GenHlth', 'Age']
    binary_features = [f for f in SELECTED_FEATURES if f not in numeric_features + ordinal_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('ord', OrdinalEncoder(dtype=np.float32), ordinal_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_data():
    """
    Полный цикл: загрузка -> сплит -> препроцессинг.
    Возвращает готовые массивы и настроенный препроцессор.
    """
    df = load_data()
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train.values, y_test.values, preprocessor