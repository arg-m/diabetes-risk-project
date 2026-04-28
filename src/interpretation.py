"""
Модуль интерпретации моделей машинного обучения
в задаче прогнозирования диабета.

Назначение:
- объяснение влияния признаков на риск заболевания
- анализ коэффициентов логистической регрессии
- определение наиболее значимых факторов риска

Результаты используются для медицинской интерпретации модели
и представления выводов пользователю.
"""

import numpy as np
import pandas as pd

from config import SELECTED_FEATURES, FEATURE_LABELS


def interpret_logistic(model):

    coef = model.coef_[0]

    print("\n=== Интерпретация риска ===")

    for f, c in zip(SELECTED_FEATURES, coef):
        label = FEATURE_LABELS[f]
        direction = "увеличивает" if c > 0 else "снижает"

        print(f"{label:30s}: {c:+.3f} -> {direction} риск")


def top_factors(model, k=3):

    coef = model.coef_[0]

    df = pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "label": [FEATURE_LABELS[f] for f in SELECTED_FEATURES],
        "coef": coef
    })

    df["abs"] = df["coef"].abs()

    return df.sort_values("abs", ascending=False).head(k)