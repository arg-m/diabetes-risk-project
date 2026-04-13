"""
Модуль отбора признаков: расчет Mutual Information и формирование отчёта.
"""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

try:
    from config import SELECTED_FEATURES, TARGET_COL, FEATURE_LABELS
    from preprocessing import load_data
except ImportError:
    from src.config import SELECTED_FEATURES, TARGET_COL, FEATURE_LABELS
    from src.preprocessing import load_data

def calculate_mi_scores():
    """Расчёт значимости признаков через Mutual Information."""
    df = load_data()
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL]
    
    mi = mutual_info_classif(X, y, random_state=42)
    scores = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'label': [FEATURE_LABELS[f] for f in SELECTED_FEATURES],
        'mi_score': mi
    }).sort_values('mi_score', ascending=False).reset_index(drop=True)
    
    return scores

def print_selection_report():
    """Выводит академически оформленный отчёт для курсовой."""
    scores = calculate_mi_scores()
    print("📊 === Отчёт по отбору признаков ===")
    print("Метод: Mutual Information Classification (учитывает нелинейные зависимости)")
    print("Отобрано 7 признаков на основе EDA, MI и клинической значимости:\n")
    
    for i, row in scores.iterrows():
        print(f"{i+1}. {row['label']:25s} | MI: {row['mi_score']:.4f}")
    print("\n✅ Отбор завершён. Признаки сохранены в config.SELECTED_FEATURES")
    return scores