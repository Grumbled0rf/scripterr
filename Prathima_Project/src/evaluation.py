"""
Model Evaluation Module
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import RESULTS_DIR


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
