"""
Модуль обучения и сравнения моделей машинного обучения
для задачи предсказания риска диабета.

Реализует:
- Logistic Regression (baseline интерпретируемая модель)
- Decision Tree (нелинейная интерпретируемая модель)
- Random Forest (ансамблевая модель повышенной устойчивости)

Использует:
- GridSearchCV для подбора гиперпараметров
- StratifiedKFold для кросс-валидации
- ROC-AUC как основную метрику качества
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import RANDOM_STATE, CV_FOLDS, SELECTED_FEATURES, FEATURE_LABELS, MODELS_DIR, MODEL_FILENAME


def get_model_and_params(model_name):
    if model_name == "logistic":
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        params = {
            "C": [0.1, 1, 10]
        }

    elif model_name == "tree":
        model = DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )
        params = {
            "max_depth": [3, 5, 7, None]
        }

    elif model_name == "rf":
        model = RandomForestClassifier(
            random_state=RANDOM_STATE
        )
        params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10]
        }

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model, params


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    model, params = get_model_and_params(model_name)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    return best_model, metrics, y_proba


def get_feature_importance(model, model_name):
    if model_name == "logistic":
        values = model.coef_[0]

    elif model_name in ["tree", "rf"]:
        values = model.feature_importances_

    else:
        return None

    df = pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "label": [FEATURE_LABELS[f] for f in SELECTED_FEATURES],
        "importance": values
    })

    return df.sort_values("importance", ascending=False)


def save_best_model(model, preprocessor, metrics, model_name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "model_name": model_name,
        "features": SELECTED_FEATURES,
        "feature_labels": FEATURE_LABELS
    }

    path = MODELS_DIR / MODEL_FILENAME

    joblib.dump(bundle, path)

    print(f"\nМодель сохранена: {path}")