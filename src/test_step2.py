import sys
from pathlib import Path
sys.path.append(str(Path("src")))

from preprocessing import prepare_data
from feature_selection import print_selection_report

print("🔄 Загрузка и предобработка данных...")
X_train, X_test, y_train, y_test, prep = prepare_data()
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Классовый баланс (train): 0={sum(y_train==0)}, 1={sum(y_train==1)}")

print("\n" + "="*50)
print_selection_report()