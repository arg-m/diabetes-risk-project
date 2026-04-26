"""
Модуль обучения, валидации и сохранения моделей.
Поддерживает LogisticRegression и DecisionTree с GridSearchCV.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from config import (
        MODELS_DIR, RANDOM_STATE, CV_FOLDS, MODEL_FILENAME,
        SELECTED_FEATURES, FEATURE_LABELS
    )
    from preprocessing import build_preprocessing_pipeline
except ImportError:
    from src.config import (
        MODELS_DIR, RANDOM_STATE, CV_FOLDS, MODEL_FILENAME,
        SELECTED_FEATURES, FEATURE_LABELS
    )
    from src.preprocessing import build_preprocessing_pipeline


def get_model_params(model_name):
    """Возвращает модель и сетку гиперпараметров для GridSearchCV."""
    if model_name == 'logistic':
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'lbfgs']
        }
    elif model_name == 'tree':
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced']
        }
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    return model, param_grid


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, preprocessor):
    """Обучение модели с GridSearchCV и оценка метрик."""
    print(f"\nОбучение {model_name.upper()}...")
    
    model, param_grid = get_model_params(model_name)
    
    # Стратифицированная кросс-валидация
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=0, return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Лучшие параметры: {grid_search.best_params_}")
    
    # Предсказания
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Метрики
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Метрики на test:")
    for name, value in metrics.items():
        print(f"  {name:10s}: {value:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    
    return best_model, metrics, cm, y_pred_proba, grid_search


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Визуализация матрицы ошибок."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Нет диабета', 'Диабет'],
        yticklabels=['Нет диабета', 'Диабет']
    )
    plt.title(f'Матрица ошибок — {model_name}')
    plt.ylabel('Фактический')
    plt.xlabel('Предсказанный')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Матрица ошибок сохранена: {save_path}")
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """Построение ROC-кривой."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-кривая — {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"ROC-кривая сохранена: {save_path}")
    plt.show()


def get_feature_importance(model, model_name):
    """Извлечение важности признаков для интерпретации."""
    if model_name == 'logistic':
        # Коэффициенты логистической регрессии (Odds Ratio)
        coef = model.coef_[0]
        importance = pd.DataFrame({
            'feature': SELECTED_FEATURES,
            'coef': coef,
            'odds_ratio': np.exp(coef),
            'label': [FEATURE_LABELS.get(f, f) for f in SELECTED_FEATURES]
        }).sort_values('coef', key=abs, ascending=False)
    elif model_name == 'tree':
        # Важность признаков из Decision Tree
        importance = pd.DataFrame({
            'feature': SELECTED_FEATURES,
            'importance': model.feature_importances_,
            'label': [FEATURE_LABELS.get(f, f) for f in SELECTED_FEATURES]
        }).sort_values('importance', ascending=False)
    else:
        importance = None
    return importance


def save_best_model(pipeline, model, metrics, model_name, output_path):
    """Сохранение пайплайна + модели + метаданных."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    package = {
        'pipeline': pipeline,
        'model': model,
        'model_name': model_name,
        'metrics': metrics,
        'features': SELECTED_FEATURES,
        'feature_labels': FEATURE_LABELS
    }
    joblib.dump(package, output_path)
    print(f"Модель сохранена: {output_path}")


def main():
    """Основной скрипт обучения."""
    from preprocessing import prepare_data
    
    # Загрузка и предобработка
    print("Загрузка данных...")
    X_train_proc, X_test_proc, y_train, y_test, preprocessor = prepare_data()
    
    # Обучение моделей
    results = {}
    for model_name in ['logistic', 'tree']:
        model, metrics, cm, y_pred_proba, grid = train_and_evaluate(
            X_train_proc, X_test_proc, y_train, y_test, model_name, preprocessor
        )
        
        # Визуализация
        plot_confusion_matrix(
            cm, model_name.upper(), 
            save_path=MODELS_DIR / f"confusion_{model_name}.png"
        )
        plot_roc_curve(
            y_test, y_pred_proba, model_name.upper(),
            save_path=MODELS_DIR / f"roc_{model_name}.png"
        )
        
        # Интерпретация
        importance = get_feature_importance(model, model_name)
        print(f"\nВажность признаков ({model_name}):")
        print(importance[['label', 'coef' if 'coef' in importance.columns else 'importance']].head(7).to_string(index=False))
        
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'importance': importance,
            'y_pred_proba': y_pred_proba
        }
    
    # Выбор лучшей модели по ROC-AUC
    best_name = max(results, key=lambda x: results[x]['metrics']['roc_auc'])
    print(f"\nЛучшая модель: {best_name.upper()} (ROC-AUC: {results[best_name]['metrics']['roc_auc']:.4f})")
    
    # Сохранение
    save_best_model(
        preprocessor, 
        results[best_name]['model'], 
        results[best_name]['metrics'],
        best_name,
        MODELS_DIR / MODEL_FILENAME
    )
    
    return results


if __name__ == '__main__':
    results = main()