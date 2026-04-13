#!/usr/bin/env python3
"""
Шаг 3 из 10: Обучение, валидация и сохранение моделей.
Запуск: python test_step3.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path("src")))

from preprocessing import prepare_data
from modeling import train_and_evaluate, save_best_model, get_feature_importance
from config import MODELS_DIR, MODEL_FILENAME

def main():
    print("🔄 Шаг 3: Загрузка и предобработка данных...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print(f"✅ Готово: Train {X_train.shape}, Test {X_test.shape}")

    print("\n📊 Обучение и оценка моделей...")
    results = {}
    
    for model_name in ['logistic', 'tree']:
        print(f"\n🔹 {model_name.upper()}")
        model, metrics, cm, y_pred_proba, _ = train_and_evaluate(
            X_train, X_test, y_train, y_test, model_name, preprocessor
        )
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'y_pred_proba': y_pred_proba
        }

        imp = get_feature_importance(model, model_name)
        col = 'coef' if 'coef' in imp.columns else 'importance'
        print("   🔑 Топ-3 признака:")
        print(imp[['label', col]].head(3).to_string(index=False))

    best_name = max(results, key=lambda x: results[x]['metrics']['roc_auc'])
    print(f"\n🏆 Лучшая модель: {best_name.upper()} (ROC-AUC: {results[best_name]['metrics']['roc_auc']:.4f})")

    save_path = MODELS_DIR / MODEL_FILENAME
    save_best_model(preprocessor, results[best_name]['model'], results[best_name]['metrics'], best_name, save_path)
    print(f"✅ Модель сохранена в {save_path}")
    print("📊 Графики (матрица ошибок, ROC-кривая) сохранены в папке models/")

if __name__ == "__main__":
    main()