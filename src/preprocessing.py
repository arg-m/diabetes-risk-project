"""
Модуль подготовки данных для модели прогнозирования диабета.

Содержит:
- загрузку датасета
- разбиение на train/test
- построение ColumnTransformer для числовых, порядковых и бинарных признаков

Не выполняет обучение моделей.
Используется как базовый слой для ML pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from config import (
    DATA_RAW,
    DATASET_FILENAME,
    TARGET_COL,
    SELECTED_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_data():
    path = DATA_RAW / DATASET_FILENAME
    df = pd.read_csv(path)

    required = SELECTED_FEATURES + [TARGET_COL]
    return df[required].copy()


def build_preprocessor():
    numeric = ["BMI"]
    ordinal = ["GenHlth", "Age"]
    binary = [f for f in SELECTED_FEATURES if f not in numeric + ordinal]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("ord", OrdinalEncoder(), ordinal),
            ("bin", "passthrough", binary),
        ]
    )


def split_data(df):
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL]

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
