#!/usr/bin/env python3
"""
Оркестратор проекта: последовательный запуск предобработки, отбора признаков, обучения и интерпретации.
Запуск: python run_pipeline.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path("src")))

from preprocessing import prepare_data
from feature_selection import calculate_mi_scores
from modeling import train_and_evaluate, save_best_model, get_feature_importance
from interpretation import print_interpretation_report
from config import MODELS_DIR, MODEL_FILENAME, SELECTED_FEATURES

def main():
    print("Шаг 1: Загрузка и предобработка данных...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    print("Шаг 2: Отбор признаков (Mutual Information)...")
    mi_df = calculate_mi_scores()
    print(mi_df.to_string(index=False))
    print()

    print("Шаг 3: Обучение моделей (LogisticRegression + DecisionTree)...")
    results = {}
    for model_name in ['logistic', 'tree']:
        print(f"\n Обучение {model_name.upper()}")
        model, metrics, cm, y_pred_proba, _ = train_and_evaluate(
            X_train, X_test, y_train, y_test, model_name, preprocessor
        )
        results[model_name] = {'model': model, 'metrics': metrics, 'y_pred_proba': y_pred_proba}
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

    best_name = max(results, key=lambda x: results[x]['metrics']['roc_auc'])
    print(f"\n Лучшая модель: {best_name.upper()}")

    print(" Шаг 4: Сохранение пайплайна и модели...")
    save_best_model(preprocessor, results[best_name]['model'], results[best_name]['metrics'], best_name, MODELS_DIR / MODEL_FILENAME)

    print("\n Шаг 5: Интерпретация...")
    print_interpretation_report(results[best_name]['model'], best_name)
    print("\n Пайплайн завершён. Модель сохранена в models/, графики готовы для отчёта.")

if __name__ == "__main__":
    main()