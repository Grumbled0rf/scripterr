"""
Model Evaluation Module
Enhanced with skill scores, statistical tests, and cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config.config import RESULTS_DIR


# =============================================================================
# BASIC METRICS
# =============================================================================

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate all evaluation metrics.

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for display

    Returns:
    --------
    dict : Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    metrics = {
        'model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Median_AE': np.median(np.abs(y_true - y_pred))
    }

    return metrics


# =============================================================================
# SKILL SCORES
# =============================================================================

def calculate_skill_score(y_true, y_pred, y_pred_baseline):
    """
    Calculate skill score relative to baseline (persistence) model.

    Skill Score = 1 - (MSE_model / MSE_baseline)
    - SS > 0: Model is better than baseline
    - SS = 0: Model equals baseline
    - SS < 0: Model is worse than baseline

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Model predictions
    y_pred_baseline : array-like
        Baseline (persistence) predictions

    Returns:
    --------
    float : Skill score
    """
    mse_model = mean_squared_error(y_true, y_pred)
    mse_baseline = mean_squared_error(y_true, y_pred_baseline)

    if mse_baseline == 0:
        return np.nan

    skill_score = 1 - (mse_model / mse_baseline)
    return skill_score


def calculate_all_skill_scores(models, X_test, y_test, persistence_predictions):
    """
    Calculate skill scores for all models vs persistence baseline.

    Returns:
    --------
    pd.DataFrame : Skill scores for each model
    """
    results = []

    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            skill = calculate_skill_score(y_test, y_pred, persistence_predictions)
            results.append({
                'model': name,
                'skill_score': skill,
                'improvement_pct': skill * 100
            })
        except Exception as e:
            results.append({
                'model': name,
                'skill_score': np.nan,
                'improvement_pct': np.nan
            })

    return pd.DataFrame(results).sort_values('skill_score', ascending=False)


# =============================================================================
# DIEBOLD-MARIANO TEST
# =============================================================================

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion='MSE'):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests the null hypothesis that two forecasts have equal accuracy.

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred1 : array-like
        Predictions from model 1
    y_pred2 : array-like
        Predictions from model 2
    h : int
        Forecast horizon (for HAC variance estimation)
    criterion : str
        Loss criterion ('MSE' or 'MAE')

    Returns:
    --------
    dict : Test statistic, p-value, and interpretation
    """
    from scipy import stats

    y_true = np.array(y_true).flatten()
    y_pred1 = np.array(y_pred1).flatten()
    y_pred2 = np.array(y_pred2).flatten()

    # Calculate loss differentials
    if criterion == 'MSE':
        e1 = (y_true - y_pred1) ** 2
        e2 = (y_true - y_pred2) ** 2
    else:  # MAE
        e1 = np.abs(y_true - y_pred1)
        e2 = np.abs(y_true - y_pred2)

    d = e1 - e2  # Loss differential
    n = len(d)

    # Calculate DM statistic
    mean_d = np.mean(d)

    # HAC variance estimator (Newey-West)
    gamma_0 = np.var(d)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1]
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        var_d = gamma_0 / n

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    # Interpretation
    if p_value < 0.05:
        if mean_d < 0:
            interpretation = "Model 1 significantly better than Model 2"
        else:
            interpretation = "Model 2 significantly better than Model 1"
    else:
        interpretation = "No significant difference between models"

    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': mean_d,
        'interpretation': interpretation
    }


def compare_models_dm(models, X_test, y_test, reference_model='XGBoost'):
    """
    Compare all models against a reference using Diebold-Mariano test.

    Returns:
    --------
    pd.DataFrame : DM test results for each model comparison
    """
    results = []

    if reference_model not in models:
        reference_model = list(models.keys())[0]

    y_pred_ref = models[reference_model].predict(X_test)

    for name, model in models.items():
        if name == reference_model:
            continue

        try:
            y_pred = model.predict(X_test)
            dm_result = diebold_mariano_test(y_test, y_pred_ref, y_pred)

            results.append({
                'comparison': f"{reference_model} vs {name}",
                'dm_statistic': dm_result['dm_statistic'],
                'p_value': dm_result['p_value'],
                'significant': dm_result['p_value'] < 0.05,
                'interpretation': dm_result['interpretation']
            })
        except Exception as e:
            results.append({
                'comparison': f"{reference_model} vs {name}",
                'dm_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'interpretation': f"Error: {str(e)}"
            })

    return pd.DataFrame(results)


# =============================================================================
# TIME SERIES CROSS-VALIDATION
# =============================================================================

def time_series_cross_validate(model_class, X, y, n_splits=5, **model_params):
    """
    Perform time series cross-validation.

    Parameters:
    -----------
    model_class : class
        Model class to instantiate
    X : pd.DataFrame
        Features
    y : np.ndarray
        Target
    n_splits : int
        Number of CV splits
    **model_params : dict
        Parameters for model instantiation

    Returns:
    --------
    dict : CV results with metrics for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}...")

        X_train_cv = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_test_cv = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
        y_train_cv = y[train_idx]
        y_test_cv = y[test_idx]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_cv, y_train_cv)

        # Predict
        y_pred = model.predict(X_test_cv)

        # Calculate metrics
        metrics = calculate_metrics(y_test_cv, y_pred, f"Fold {fold + 1}")
        metrics['fold'] = fold + 1
        metrics['train_size'] = len(train_idx)
        metrics['test_size'] = len(test_idx)

        fold_results.append(metrics)

    results_df = pd.DataFrame(fold_results)

    # Summary statistics
    summary = {
        'mean_MAE': results_df['MAE'].mean(),
        'std_MAE': results_df['MAE'].std(),
        'mean_RMSE': results_df['RMSE'].mean(),
        'std_RMSE': results_df['RMSE'].std(),
        'mean_MAPE': results_df['MAPE'].mean(),
        'std_MAPE': results_df['MAPE'].std(),
        'mean_R2': results_df['R2'].mean(),
        'std_R2': results_df['R2'].std()
    }

    return {
        'fold_results': results_df,
        'summary': summary
    }


def run_cv_for_all_models(X, y, n_splits=5):
    """
    Run time series CV for multiple model types.

    Returns:
    --------
    dict : CV results for each model type
    """
    from src.models import RandomForestModel, XGBoostModel

    print("\n" + "=" * 50)
    print("TIME SERIES CROSS-VALIDATION")
    print("=" * 50)

    results = {}

    # Random Forest CV
    print("\nRandom Forest CV...")
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf_cv = time_series_cross_validate(
            RandomForestRegressor,
            X, y, n_splits,
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        results['Random Forest'] = rf_cv
        print(f"  Mean RMSE: {rf_cv['summary']['mean_RMSE']:.2f} ± {rf_cv['summary']['std_RMSE']:.2f}")
    except Exception as e:
        print(f"  Error: {e}")

    # XGBoost CV
    print("\nXGBoost CV...")
    try:
        from xgboost import XGBRegressor
        xgb_cv = time_series_cross_validate(
            XGBRegressor,
            X, y, n_splits,
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        results['XGBoost'] = xgb_cv
        print(f"  Mean RMSE: {xgb_cv['summary']['mean_RMSE']:.2f} ± {xgb_cv['summary']['std_RMSE']:.2f}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


# =============================================================================
# PREDICTION INTERVAL EVALUATION
# =============================================================================

def evaluate_prediction_intervals(y_true, y_lower, y_upper, confidence=0.90):
    """
    Evaluate prediction interval quality.

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_lower : array-like
        Lower bound predictions
    y_upper : array-like
        Upper bound predictions
    confidence : float
        Expected coverage (e.g., 0.90 for 90% PI)

    Returns:
    --------
    dict : Interval evaluation metrics
    """
    y_true = np.array(y_true).flatten()
    y_lower = np.array(y_lower).flatten()
    y_upper = np.array(y_upper).flatten()

    # Coverage: proportion of actual values within interval
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = np.mean(in_interval)

    # Average interval width
    interval_width = y_upper - y_lower
    avg_width = np.mean(interval_width)
    avg_width_pct = np.mean(interval_width / y_true) * 100

    # Interval score (proper scoring rule)
    alpha = 1 - confidence
    interval_score = np.mean(
        (y_upper - y_lower) +
        (2 / alpha) * (y_lower - y_true) * (y_true < y_lower) +
        (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
    )

    return {
        'coverage': coverage,
        'expected_coverage': confidence,
        'coverage_error': coverage - confidence,
        'avg_interval_width': avg_width,
        'avg_interval_width_pct': avg_width_pct,
        'interval_score': interval_score
    }


# =============================================================================
# ORIGINAL EVALUATION FUNCTIONS
# =============================================================================

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on test data.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : np.ndarray
        Test target

    Returns:
    --------
    pd.DataFrame : Comparison of all models
    """
    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        try:
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, name)
            results.append(metrics)

            print(f"  MAE: {metrics['MAE']:.2f}")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²: {metrics['R2']:.4f}")

        except Exception as e:
            print(f"  Error evaluating {name}: {e}")
            results.append({
                'model': name,
                'MAE': np.nan,
                'RMSE': np.nan,
                'MAPE': np.nan,
                'R2': np.nan,
                'Max_Error': np.nan,
                'Median_AE': np.nan
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')

    return results_df


def evaluate_extreme_events(model, X_test, y_test, df_test, threshold_percentile=95):
    """
    Evaluate model performance during extreme demand events.

    Parameters:
    -----------
    model : trained model
        Model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : np.ndarray
        Test target
    df_test : pd.DataFrame
        Test dataframe with datetime index
    threshold_percentile : float
        Percentile threshold for extreme events

    Returns:
    --------
    dict : Metrics for extreme events
    """
    y_pred = model.predict(X_test)

    # Find extreme demand periods (high demand)
    high_demand_threshold = np.percentile(y_test, threshold_percentile)
    high_demand_mask = y_test >= high_demand_threshold

    # Find low demand periods
    low_demand_threshold = np.percentile(y_test, 100 - threshold_percentile)
    low_demand_mask = y_test <= low_demand_threshold

    results = {
        'normal_conditions': calculate_metrics(
            y_test[~high_demand_mask & ~low_demand_mask],
            y_pred[~high_demand_mask & ~low_demand_mask],
            "Normal"
        ),
        'high_demand': calculate_metrics(
            y_test[high_demand_mask],
            y_pred[high_demand_mask],
            "High Demand"
        ),
        'low_demand': calculate_metrics(
            y_test[low_demand_mask],
            y_pred[low_demand_mask],
            "Low Demand"
        )
    }

    print("\n" + "=" * 50)
    print("EXTREME EVENT ANALYSIS")
    print("=" * 50)

    for condition, metrics in results.items():
        print(f"\n{condition.upper()}:")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")

    return results


def evaluate_by_time_period(model, X_test, y_test, df_test):
    """
    Evaluate model performance by different time periods.

    Parameters:
    -----------
    model : trained model
        Model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : np.ndarray
        Test target
    df_test : pd.DataFrame
        Test dataframe with datetime index

    Returns:
    --------
    dict : Metrics by time period
    """
    y_pred = model.predict(X_test)

    # Ensure we have datetime index
    if hasattr(df_test.index, 'hour'):
        hours = df_test.index.hour
        months = df_test.index.month
        day_of_week = df_test.index.dayofweek
    else:
        return {}

    results = {}

    # By hour of day
    hourly_metrics = []
    for hour in range(24):
        mask = hours == hour
        if mask.sum() > 0:
            metrics = calculate_metrics(y_test[mask], y_pred[mask], f"Hour {hour}")
            metrics['hour'] = hour
            hourly_metrics.append(metrics)
    results['hourly'] = pd.DataFrame(hourly_metrics)

    # By day of week
    daily_metrics = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_idx, day_name in enumerate(days):
        mask = day_of_week == day_idx
        if mask.sum() > 0:
            metrics = calculate_metrics(y_test[mask], y_pred[mask], day_name)
            metrics['day'] = day_name
            daily_metrics.append(metrics)
    results['daily'] = pd.DataFrame(daily_metrics)

    # By month
    monthly_metrics = []
    for month in range(1, 13):
        mask = months == month
        if mask.sum() > 0:
            metrics = calculate_metrics(y_test[mask], y_pred[mask], f"Month {month}")
            metrics['month'] = month
            monthly_metrics.append(metrics)
    results['monthly'] = pd.DataFrame(monthly_metrics)

    # Weekend vs Weekday
    weekend_mask = day_of_week >= 5
    results['weekend'] = calculate_metrics(y_test[weekend_mask], y_pred[weekend_mask], "Weekend")
    results['weekday'] = calculate_metrics(y_test[~weekend_mask], y_pred[~weekend_mask], "Weekday")

    return results


def generate_evaluation_report(models, X_test, y_test, df_test, save_path=None):
    """
    Generate comprehensive evaluation report.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : np.ndarray
        Test target
    df_test : pd.DataFrame
        Test dataframe with datetime index
    save_path : Path, optional
        Path to save report

    Returns:
    --------
    dict : Complete evaluation results
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 60)

    report = {}

    # Overall metrics
    print("\n" + "-" * 40)
    print("OVERALL MODEL COMPARISON")
    print("-" * 40)
    report['overall'] = evaluate_models(models, X_test, y_test)
    print("\n" + report['overall'].to_string(index=False))

    # Best model analysis
    best_model_name = report['overall'].iloc[0]['model']
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    # Extreme events for best model
    print("\n" + "-" * 40)
    print(f"EXTREME EVENT ANALYSIS ({best_model_name})")
    print("-" * 40)
    report['extreme_events'] = evaluate_extreme_events(best_model, X_test, y_test, df_test)

    # Time period analysis for best model
    print("\n" + "-" * 40)
    print(f"TIME PERIOD ANALYSIS ({best_model_name})")
    print("-" * 40)
    report['time_periods'] = evaluate_by_time_period(best_model, X_test, y_test, df_test)

    if 'hourly' in report['time_periods']:
        print("\nWorst performing hours:")
        hourly = report['time_periods']['hourly'].sort_values('MAPE', ascending=False)
        print(hourly.head(5)[['hour', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))

    # Save report
    if save_path:
        save_path = Path(save_path)
        report['overall'].to_csv(save_path / 'model_comparison.csv', index=False)

        if 'hourly' in report['time_periods']:
            report['time_periods']['hourly'].to_csv(save_path / 'hourly_performance.csv', index=False)
        if 'monthly' in report['time_periods']:
            report['time_periods']['monthly'].to_csv(save_path / 'monthly_performance.csv', index=False)

        print(f"\nReport saved to {save_path}")

    return report
