from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
DATASET_FILENAME = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
TARGET_COL = "Diabetes_binary"
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_FILENAME = "best_pipeline.joblib"

SELECTED_FEATURES = [
    'GenHlth',              # Общее состояние здоровья (ordinal 1-5)
    'BMI',                  # Индекс массы тела (continuous)
    'Age',                  # Возраст (ordinal 1-13)
    'HighBP',               # Гипертония (binary)
    'PhysActivity',         # Физическая активность (binary)
    'HeartDiseaseorAttack', # Сердечно-сосудистые заболевания (binary)
    'DiffWalk'              # Трудности при ходьбе (binary)
]

FEATURE_LABELS = {
    'GenHlth': 'Общее состояние здоровья',
    'BMI': 'ИМТ',
    'Age': 'Возрастная группа',
    'HighBP': 'Высокое кровяное давление',
    'PhysActivity': 'Физическая активность',
    'HeartDiseaseorAttack': 'Болезни сердца/инфаркт',
    'DiffWalk': 'Трудности при ходьбе'
}

CATEGORY_MAPS = {
    'Age': {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
            6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
            11: '70-74', 12: '75-79', 13: '≥80'},
    'GenHlth': {1: 'Отличное', 2: 'Очень хорошее', 3: 'Хорошее',
                4: 'Удовлетворительное', 5: 'Плохое'}
}
