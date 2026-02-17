# Project Improvements Roadmap

## Priority 1: Critical for Dissertation

### 1.1 Add Statistical Baseline (ARIMA)
```python
# Add to src/models.py
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
```
- Implement SARIMA(1,1,1)(1,1,1,24) for hourly seasonality
- Compare with ML models to show improvement

### 1.2 Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train and evaluate
```

### 1.3 Add Persistence Baseline
```python
# Naive forecast: predict previous hour's value
y_pred_persistence = y_test_shifted_1h
# Calculate skill score vs persistence
skill_score = 1 - (MSE_model / MSE_persistence)
```

### 1.4 Statistical Significance Test
```python
# Diebold-Mariano test for forecast comparison
from statsmodels.stats.diagnostic import acorr_ljungbox
# Compare XGBoost vs Random Forest predictions
```

---

## Priority 2: Methodology Enhancement

### 2.1 Weather Data Integration
```python
# OPSD includes weather data
weather_cols = [
    'IE_temperature',  # If available
    'IE_wind_onshore_generation_actual',  # Proxy for wind
    'IE_solar_generation_actual'  # Proxy for solar irradiance
]
```

### 2.2 Multiple Forecast Horizons
- 1-hour ahead (current)
- 6-hour ahead
- 24-hour ahead (day-ahead market)
- 168-hour ahead (week-ahead)

### 2.3 Prediction Intervals
```python
# Quantile regression for prediction intervals
from sklearn.ensemble import GradientBoostingRegressor

# Train models for 5th, 50th, 95th percentiles
model_q05 = GradientBoostingRegressor(loss='quantile', alpha=0.05)
model_q50 = GradientBoostingRegressor(loss='quantile', alpha=0.50)
model_q95 = GradientBoostingRegressor(loss='quantile', alpha=0.95)
```

---

## Priority 3: Additional Analysis

### 3.1 Ablation Study
Test model performance by removing feature groups:
1. Without lag features
2. Without rolling features
3. Without temporal features
4. Without holiday features

### 3.2 Feature Importance Comparison
| Method | Top Feature |
|--------|-------------|
| XGBoost native | is_night |
| SHAP | demand_lag_1h |
| Permutation | ? |

### 3.3 Error Analysis by Period
- COVID-19 impact (Mar-Sep 2020)
- Holiday periods
- Extreme weather events

---

## Priority 4: Code Quality

### 4.1 Add Unit Tests
```
tests/
├── test_data_loader.py
├── test_feature_engineering.py
├── test_models.py
└── test_evaluation.py
```

### 4.2 Add Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### 4.3 Configuration Management
```yaml
# config.yaml
model:
  xgboost:
    n_estimators: 500
    max_depth: 8
data:
  source: "OPSD"
  version: "2020-10-06"
```

---

## Priority 5: Visualization Enhancements

### 5.1 Missing Plots
- [ ] Residual autocorrelation plot (ACF/PACF)
- [ ] Actual vs Predicted by hour heatmap
- [ ] Prediction intervals visualization
- [ ] Learning curves (performance vs training size)
- [ ] Feature correlation matrix

### 5.2 Interactive Dashboard
```python
# Optional: Streamlit dashboard
import streamlit as st
# Real-time predictions and visualizations
```

---

## Comparison Table Template (For Dissertation)

| Study | Model | Data | MAPE | RMSE | R² |
|-------|-------|------|------|------|-----|
| Quansah & Tenkorang (2023) | CNN-LSTM | Private | 1.94% | - | - |
| Liu et al. (2023) | SaDI | Private | - | -22% vs baseline | - |
| Lotfi et al. (2025) | Random Forest | Private | - | - | 0.894 |
| **This Study** | **XGBoost** | **OPSD (Open)** | **0.20%** | **13.50 MW** | **0.9994** |

---

## Implementation Checklist

- [ ] SARIMA baseline model
- [ ] Time series cross-validation
- [ ] Persistence model baseline
- [ ] Skill scores calculation
- [ ] Weather data integration
- [ ] Multiple forecast horizons
- [ ] Prediction intervals
- [ ] Ablation study
- [ ] Diebold-Mariano test
- [ ] Residual diagnostics
- [ ] COVID-19 period analysis
- [ ] Unit tests
- [ ] Documentation improvements
