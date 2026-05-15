"""
Модуль обучения, оценки и сравнения моделей машинного обучения
для задачи прогнозирования риска диабета.

Реализует:
- Logistic Regression
- Decision Tree
- Random Forest

Поддерживает:
- GridSearchCV
- Stratified K-Fold Cross Validation
- ROC-AUC как основную метрику
- Расчёт метрик качества
- ROC-кривые
- Confusion Matrix
- Feature Importance
- Сравнение моделей

Архитектурная идея:
Вся ML-логика находится в src/,
а notebooks используются только для анализа и визуализации.
"""

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    SELECTED_FEATURES,
    FEATURE_LABELS,
    MODELS_DIR,
    MODEL_FILENAME
)


def get_model_and_params(model_name):
    """
    Возвращает модель и сетку гиперпараметров.
    """

    if model_name == "logistic":

        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )

        params = {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"]
        }

    elif model_name == "tree":

        model = DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )

        params = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [1, 5, 10],
            "min_samples_split": [2, 5, 10]
        }

    elif model_name == "rf":

        model = RandomForestClassifier(
            random_state=RANDOM_STATE
        )

        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model, params


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name
):
    """
    Обучение модели и расчёт метрик.
    """

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

    print(f"{model_name} best params: {grid.best_params_}")

    y_pred = best_model.predict(X_test)

    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    return best_model, metrics, y_pred, y_proba


def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    model_names
):
    """
    Сравнение нескольких моделей.
    """

    rows = []

    trained_models = {}

    roc_data = {}

    for model_name in model_names:

        model, metrics, y_pred, y_proba = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            model_name
        )

        trained_models[model_name] = model

        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"]
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        roc_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": metrics["roc_auc"]
        }

    results_df = pd.DataFrame(rows)

    return results_df, trained_models, roc_data


def cross_validate_models(X, y, trained_models):
    """
    Cross-validation анализ на уже подобранных моделях.
    """
    rows = []

    for model_name, model in trained_models.items():

        scores = cross_val_score(
            model,
            X,
            y,
            cv=CV_FOLDS,
            scoring="roc_auc"
        )

        rows.append({
            "model": model_name,
            "cv_mean": scores.mean(),
            "cv_std": scores.std()
        })

    return pd.DataFrame(rows)


def get_feature_importance(
    model,
    model_name
):
    """
    Получение важности признаков.
    """

    if model_name == "logistic":

        values = np.abs(model.coef_[0])

    elif model_name in ["tree", "rf"]:

        values = model.feature_importances_

    else:
        return None

    df = pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "label": [FEATURE_LABELS[f] for f in SELECTED_FEATURES],
        "importance": values
    })

    return df.sort_values(
        "importance",
        ascending=False
    )


def plot_roc_curves(roc_data):
    """
    Построение ROC-кривых моделей.
    """

    plt.figure(figsize=(8, 6))

    for model_name, data in roc_data.items():

        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{model_name} (AUC = {data['auc']:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC-кривые моделей")

    plt.legend()

    plt.show()


def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name
):
    """
    Построение матрицы ошибок.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix — {model_name}")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()


def save_best_model(
    model,
    preprocessor,
    metrics,
    model_name
):
    """
    Сохранение лучшей модели.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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