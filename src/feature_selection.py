"""
Модуль анализа значимости признаков с использованием Mutual Information.

Назначение:
- оценка информативности признаков относительно целевой переменной
- формирование ранжированного списка факторов риска диабета

Важно:
данный модуль носит аналитический характер и не влияет напрямую на обучение моделей.
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import SELECTED_FEATURES, FEATURE_LABELS


def calculate_mi_scores(X_train, y_train):
    mi = mutual_info_classif(X_train, y_train, random_state=42)

    df = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'label': [FEATURE_LABELS[f] for f in SELECTED_FEATURES],
        'mi_score': mi
    }).sort_values('mi_score', ascending=False)

    return df


def print_mi_report(df):
    print("\n=== Mutual Information ===")
    for i, row in df.iterrows():
        print(f"{i+1}. {row['label']} -> {row['mi_score']:.4f}")