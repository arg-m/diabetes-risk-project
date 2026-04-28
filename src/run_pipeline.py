"""
Главный оркестратор ML-пайплайна проекта прогнозирования диабета.

Последовательно выполняет:
1. загрузку и подготовку данных
2. анализ значимости признаков
3. обучение нескольких моделей
4. сравнение моделей по ROC-AUC
5. интерпретацию лучшей модели

Является точкой входа для полного ML-процесса.
"""

from preprocessing import load_data, build_preprocessor, split_data
from feature_selection import calculate_mi_scores, print_mi_report
from modeling import train_and_evaluate, save_best_model


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    mi = calculate_mi_scores(X_train, y_train)
    print_mi_report(mi)

    preprocessor = build_preprocessor()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    results = {}

    for model_name in ["logistic", "tree", "rf"]:
        print(f"\nОбучение модели: {model_name}")

        model, metrics, y_proba = train_and_evaluate(
            X_train_proc, X_test_proc, y_train, y_test, model_name
        )

        results[model_name] = {"model": model, "metrics": metrics}

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    best_name = max(results, key=lambda x: results[x]["metrics"]["roc_auc"])

    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    print(f"\nЛучшая модель: {best_name.upper()}")

    save_best_model(
        model=best_model,
        preprocessor=preprocessor,
        metrics=best_metrics,
        model_name=best_name,
    )

    print("\nPipeline завершён успешно.")


if __name__ == "__main__":
    main()
