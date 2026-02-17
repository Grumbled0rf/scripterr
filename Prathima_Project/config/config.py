"""
Configuration file for Ireland Electricity Demand Forecasting Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data URLs - Open Power System Data (2020-10-06 version)
OPSD_TIME_SERIES_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

# Data parameters
COUNTRY = "IE"  # Ireland
TARGET_COLUMN = "IE_load_actual_entsoe_transparency"
DATE_COLUMN = "utc_timestamp"

# Feature engineering parameters
LAG_FEATURES = [1, 2, 3, 24, 48, 168]  # 1h, 2h, 3h, 24h, 48h, 1 week
ROLLING_WINDOWS = [6, 12, 24, 48, 168]  # Rolling statistics windows

# Model parameters
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# LSTM parameters
LSTM_SEQUENCE_LENGTH = 168  # 1 week of hourly data
LSTM_UNITS = [128, 64]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 10  # Early stopping patience

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Evaluation metrics
METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']
