"""
Модуль интерпретации моделей: коэффициенты, Odds Ratio, важность признаков.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from config import FEATURE_LABELS, SELECTED_FEATURES
except ImportError:
    from src.config import FEATURE_LABELS, SELECTED_FEATURES


def plot_logistic_coefficients(model, save_path=None):
    """Визуализация коэффициентов логистической регрессии с доверительными интервалами."""
    coef = model.coef_[0]
    std_err = np.sqrt(np.diag(model.covariance_)) if hasattr(model, 'covariance_') else None
    
    features = [FEATURE_LABELS.get(f, f) for f in SELECTED_FEATURES]
    y_pos = np.arange(len(features))
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(y_pos, coef, color=['#2D8730' if c > 0 else '#9E2A22' for c in coef])
    
    if std_err is not None:
        plt.errorbar(coef, y_pos, xerr=1.96*std_err, fmt='none', ecolor='gray', capsize=3, alpha=0.5)
    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.yticks(y_pos, features)
    plt.xlabel('Коэффициент (log-odds)')
    plt.title('Влияние признаков на риск диабета (Logistic Regression)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_odds_ratio(model, save_path=None):
    """Визуализация Odds Ratio с 95% доверительным интервалом."""
    coef = model.coef_[0]
    odds_ratio = np.exp(coef)
    
    # Приближённый 95% CI
    if hasattr(model, 'covariance_'):
        std_err = np.sqrt(np.diag(model.covariance_))
        ci_lower = np.exp(coef - 1.96 * std_err)
        ci_upper = np.exp(coef + 1.96 * std_err)
    else:
        ci_lower = ci_upper = odds_ratio
    
    features = [FEATURE_LABELS.get(f, f) for f in SELECTED_FEATURES]
    y_pos = np.arange(len(features))
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        odds_ratio, y_pos, 
        xerr=[odds_ratio - ci_lower, ci_upper - odds_ratio],
        fmt='o', capsize=5, ecolor='gray', alpha=0.7
    )
    plt.axvline(1, color='red', linestyle='--', label='OR = 1 (нейтрально)')
    plt.yticks(y_pos, features)
    plt.xlabel('Odds Ratio (экспонента коэффициента)')
    plt.title('Отношение шансов для риска диабета')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_interpretation_report(model, model_name):
    print(f"\nИНТЕРПРЕТАЦИЯ МОДЕЛИ: {model_name.upper()}")
    print("=" * 60)
    
    if model_name == 'logistic':
        coef = model.coef_[0]
        odds = np.exp(coef)
        for feat, c, o in zip(SELECTED_FEATURES, coef, odds):
            label = FEATURE_LABELS.get(feat, feat)
            direction = "увеличивает" if c > 0 else "снижает"
            print(f"• {label:25s}: log-odds = {c:+.3f}, OR = {o:.2f} → {direction} риск")
    
    elif model_name == 'tree':
        imp = pd.DataFrame({
            'feature': SELECTED_FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        for _, row in imp.head(5).iterrows():
            label = FEATURE_LABELS.get(row['feature'], row['feature'])
            print(f"• {label:25s}: важность = {row['importance']:.3f}")
    
    print("=" * 60)