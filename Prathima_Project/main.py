"""
Main Execution Script for Ireland Electricity Demand Forecasting
=================================================================

Hybrid Interpretable Machine Learning Model for Hourly Electricity Demand
Forecasting in Ireland using Open Power System Data (OPSD)

This script implements:
1. Data loading from OPSD
2. Feature engineering (temporal, lag, rolling features)
3. Multiple ML models (Random Forest, XGBoost, LSTM, Hybrid)
4. SHAP-based explainability
5. Comprehensive evaluation

Author: Prathima Project
"""

import sys
import warnings
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# Import project modules
from config.config import (
    DATA_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR,
    TRAIN_TEST_SPLIT, VALIDATION_SPLIT
)
from src.data_loader import load_and_prepare_data
from src.feature_engineering import engineer_features, get_feature_columns
from src.preprocessing import preprocess_pipeline, split_data, handle_missing_values
from src.models import (
    RandomForestModel, XGBoostModel, LSTMModel, HybridModel,
    train_all_models
)
from src.evaluation import (
    evaluate_models, evaluate_extreme_events,
    evaluate_by_time_period, generate_evaluation_report
)
from src.visualization import (
    create_eda_report, plot_predictions, plot_model_comparison,
    plot_feature_importance
)
from src.explainability import generate_shap_report


def run_data_pipeline(force_download=False):
    """
    Execute data loading and preparation pipeline.

    Returns:
    --------
    pd.DataFrame : Prepared data with features
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 70)

    # Load raw data
    df = load_and_prepare_data(force_download=force_download)

    # Engineer features
    print("\n" + "-" * 50)
    df = engineer_features(df)

    # Save processed data
    processed_path = DATA_DIR / "ireland_demand_featured.csv"
    df.to_csv(processed_path)
    print(f"\nFeatured data saved to {processed_path}")

    return df


def run_eda(df):
    """
    Execute exploratory data analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Prepared data with features
    """
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    create_eda_report(df, save_dir=PLOTS_DIR)


def prepare_train_test_data(df):
    """
    Prepare train, validation, and test datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        Featured data

    Returns:
    --------
    dict : Dictionary containing data splits and metadata
    """
    print("\n" + "=" * 70)
    print("STEP 3: DATA SPLITTING AND PREPARATION")
    print("=" * 70)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nNumber of features: {len(feature_cols)}")

    # Split data
    train_df, val_df, test_df = split_data(df)

    # Prepare X and y
    X_train = train_df[feature_cols]
    y_train = train_df['demand'].values

    X_val = val_df[feature_cols]
    y_val = val_df['demand'].values

    X_test = test_df[feature_cols]
    y_test = test_df['demand'].values

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'feature_columns': feature_cols
    }


def train_models(data_dict):
    """
    Train all machine learning models.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data splits

    Returns:
    --------
    dict : Dictionary of trained models
    """
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)

    models = {}

    # Extract data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    feature_columns = data_dict['feature_columns']

    # 1. Random Forest
    print("\n" + "-" * 50)
    print("Training Random Forest...")
    print("-" * 50)
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model

    # 2. XGBoost
    print("\n" + "-" * 50)
    print("Training XGBoost...")
    print("-" * 50)
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb_model

    # 3. Hybrid Model (XGBoost + LSTM)
    print("\n" + "-" * 50)
    print("Training Hybrid Model (XGBoost + LSTM)...")
    print("-" * 50)
    try:
        hybrid_model = HybridModel()
        hybrid_model.fit(X_train, y_train, X_val, y_val, feature_columns)
        models['Hybrid'] = hybrid_model
    except Exception as e:
        print(f"Warning: Hybrid model training failed: {e}")
        print("Continuing without hybrid model...")

    # Save models
    print("\n" + "-" * 50)
    print("Saving models...")
    print("-" * 50)

    for name, model in models.items():
        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_model.pkl"
        if name == 'Hybrid':
            model.save(MODELS_DIR / 'hybrid_model')
        else:
            model.save(model_path)
        print(f"  Saved: {name}")

    return models


def evaluate_all_models(models, data_dict):
    """
    Evaluate all trained models.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    data_dict : dict
        Dictionary containing data splits

    Returns:
    --------
    dict : Evaluation results
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    test_df = data_dict['test_df']

    # Generate comprehensive evaluation report
    report = generate_evaluation_report(
        models, X_test, y_test, test_df,
        save_path=RESULTS_DIR
    )

    # Plot model comparison
    if 'overall' in report:
        plot_model_comparison(
            report['overall'],
            save_path=PLOTS_DIR / 'model_comparison.png'
        )

    # Plot predictions for best model
    best_model_name = report['overall'].iloc[0]['model']
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)

    plot_predictions(
        y_test, y_pred,
        dates=test_df.index,
        model_name=best_model_name,
        save_path=PLOTS_DIR / f'{best_model_name.lower()}_predictions.png'
    )

    # Plot feature importance
    if hasattr(best_model, 'get_feature_importance'):
        importance = best_model.get_feature_importance()
        plot_feature_importance(
            importance,
            title=f'{best_model_name} Feature Importance',
            save_path=PLOTS_DIR / f'{best_model_name.lower()}_feature_importance.png'
        )

    return report


def run_explainability(models, data_dict):
    """
    Run SHAP explainability analysis.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    data_dict : dict
        Dictionary containing data splits

    Returns:
    --------
    dict : SHAP analysis results
    """
    print("\n" + "=" * 70)
    print("STEP 6: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    X_test = data_dict['X_test']
    feature_columns = data_dict['feature_columns']

    # Use XGBoost for SHAP (best compatibility)
    if 'XGBoost' in models:
        model = models['XGBoost']
    elif 'Random Forest' in models:
        model = models['Random Forest']
    else:
        print("No suitable model for SHAP analysis")
        return {}

    try:
        shap_results = generate_shap_report(
            model,
            X_test,
            feature_columns,
            save_dir=PLOTS_DIR / 'shap'
        )
        return shap_results
    except Exception as e:
        print(f"Warning: SHAP analysis failed: {e}")
        return {}


def generate_final_report(evaluation_report, shap_results, data_dict):
    """
    Generate final summary report.

    Parameters:
    -----------
    evaluation_report : dict
        Evaluation results
    shap_results : dict
        SHAP analysis results
    data_dict : dict
        Data information
    """
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)

    # Data summary
    print("\n" + "-" * 50)
    print("DATA SUMMARY")
    print("-" * 50)
    print(f"Training samples: {len(data_dict['X_train'])}")
    print(f"Validation samples: {len(data_dict['X_val'])}")
    print(f"Test samples: {len(data_dict['X_test'])}")
    print(f"Number of features: {len(data_dict['feature_columns'])}")
    print(f"Date range: {data_dict['train_df'].index.min()} to {data_dict['test_df'].index.max()}")

    # Model performance summary
    print("\n" + "-" * 50)
    print("MODEL PERFORMANCE SUMMARY")
    print("-" * 50)
    if 'overall' in evaluation_report:
        print(evaluation_report['overall'].to_string(index=False))

    # Best model
    if 'overall' in evaluation_report:
        best = evaluation_report['overall'].iloc[0]
        print(f"\nBest Model: {best['model']}")
        print(f"  RMSE: {best['RMSE']:.2f} MW")
        print(f"  MAPE: {best['MAPE']:.2f}%")
        print(f"  R²: {best['R2']:.4f}")

    # Top features
    if shap_results and 'importance' in shap_results:
        print("\n" + "-" * 50)
        print("TOP 10 IMPORTANT FEATURES (SHAP)")
        print("-" * 50)
        print(shap_results['importance'].head(10).to_string(index=False))

    # Save summary
    summary_path = RESULTS_DIR / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IRELAND ELECTRICITY DEMAND FORECASTING - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Data Summary:\n")
        f.write(f"  Training samples: {len(data_dict['X_train'])}\n")
        f.write(f"  Validation samples: {len(data_dict['X_val'])}\n")
        f.write(f"  Test samples: {len(data_dict['X_test'])}\n")
        f.write(f"  Number of features: {len(data_dict['feature_columns'])}\n\n")

        if 'overall' in evaluation_report:
            f.write("Model Performance:\n")
            f.write(evaluation_report['overall'].to_string(index=False))

    print(f"\nSummary report saved to {summary_path}")


def main(force_download=False, skip_eda=False, skip_shap=False):
    """
    Main execution function.

    Parameters:
    -----------
    force_download : bool
        Force re-download of data
    skip_eda : bool
        Skip EDA visualizations
    skip_shap : bool
        Skip SHAP analysis
    """
    print("\n" + "=" * 70)
    print("IRELAND ELECTRICITY DEMAND FORECASTING")
    print("Hybrid Interpretable Machine Learning Model")
    print("=" * 70)

    # Step 1: Data Loading and Preparation
    df = run_data_pipeline(force_download)

    # Step 2: EDA
    if not skip_eda:
        run_eda(df)

    # Step 3: Prepare train/test data
    data_dict = prepare_train_test_data(df)

    # Step 4: Train models
    models = train_models(data_dict)

    # Step 5: Evaluate models
    evaluation_report = evaluate_all_models(models, data_dict)

    # Step 6: SHAP Explainability
    if not skip_shap:
        shap_results = run_explainability(models, data_dict)
    else:
        shap_results = {}

    # Final Report
    generate_final_report(evaluation_report, shap_results, data_dict)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")

    return {
        'models': models,
        'evaluation': evaluation_report,
        'shap': shap_results,
        'data': data_dict
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ireland Electricity Demand Forecasting')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of OPSD data')
    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip exploratory data analysis')
    parser.add_argument('--skip-shap', action='store_true',
                       help='Skip SHAP explainability analysis')

    args = parser.parse_args()

    results = main(
        force_download=args.force_download,
        skip_eda=args.skip_eda,
        skip_shap=args.skip_shap
    )
