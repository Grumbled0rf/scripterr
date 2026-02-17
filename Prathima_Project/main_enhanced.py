"""
Enhanced Main Execution Script for Ireland Electricity Demand Forecasting
==========================================================================

Includes all improvements:
1. SARIMA baseline model
2. Time-series cross-validation
3. Weather features from OPSD
4. Prediction intervals
5. Ablation study
6. Persistence baseline and skill scores
7. Diebold-Mariano statistical test

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
from src.preprocessing import split_data
from src.models import (
    PersistenceModel, SARIMAModel,
    RandomForestModel, XGBoostModel, LightGBMModel,
    QuantileRegressionModel
)
from src.evaluation import (
    evaluate_models, calculate_metrics,
    calculate_skill_score, calculate_all_skill_scores,
    diebold_mariano_test, compare_models_dm,
    run_cv_for_all_models,
    evaluate_prediction_intervals,
    evaluate_extreme_events, evaluate_by_time_period,
    generate_evaluation_report
)
from src.visualization import (
    create_eda_report, plot_predictions, plot_model_comparison,
    plot_feature_importance
)
from src.explainability import generate_shap_report
from src.ablation_study import generate_ablation_report


def run_enhanced_pipeline(force_download=False, skip_eda=False, skip_shap=False,
                          skip_sarima=False, skip_cv=False, skip_ablation=False):
    """
    Run the enhanced forecasting pipeline with all improvements.
    """
    print("\n" + "=" * 70)
    print("IRELAND ELECTRICITY DEMAND FORECASTING")
    print("Enhanced Pipeline with Statistical Tests & Ablation Study")
    print("=" * 70)

    results = {}

    # =========================================================================
    # STEP 1: DATA LOADING AND PREPARATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 70)

    df = load_and_prepare_data(force_download=force_download)
    df = engineer_features(df)

    # Save processed data
    processed_path = DATA_DIR / "ireland_demand_featured.csv"
    df.to_csv(processed_path)
    print(f"\nFeatured data saved to {processed_path}")

    # =========================================================================
    # STEP 2: EDA (Optional)
    # =========================================================================
    if not skip_eda:
        print("\n" + "=" * 70)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        create_eda_report(df, save_dir=PLOTS_DIR)

    # =========================================================================
    # STEP 3: DATA SPLITTING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: DATA SPLITTING")
    print("=" * 70)

    feature_cols = get_feature_columns(df)
    print(f"Number of features: {len(feature_cols)}")

    train_df, val_df, test_df = split_data(df)

    X_train = train_df[feature_cols]
    y_train = train_df['demand'].values

    X_val = val_df[feature_cols]
    y_val = val_df['demand'].values

    X_test = test_df[feature_cols]
    y_test = test_df['demand'].values

    results['data'] = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'feature_columns': feature_cols
    }

    # =========================================================================
    # STEP 4: TIME SERIES CROSS-VALIDATION (Optional)
    # =========================================================================
    if not skip_cv:
        print("\n" + "=" * 70)
        print("STEP 4: TIME SERIES CROSS-VALIDATION")
        print("=" * 70)

        cv_results = run_cv_for_all_models(X_train, y_train, n_splits=5)
        results['cv'] = cv_results

        # Save CV results
        cv_summary = []
        for model_name, cv_data in cv_results.items():
            cv_summary.append({
                'model': model_name,
                'mean_RMSE': cv_data['summary']['mean_RMSE'],
                'std_RMSE': cv_data['summary']['std_RMSE'],
                'mean_R2': cv_data['summary']['mean_R2']
            })
        cv_df = pd.DataFrame(cv_summary)
        cv_df.to_csv(RESULTS_DIR / 'cross_validation_results.csv', index=False)
        print(f"\nCV results saved to {RESULTS_DIR / 'cross_validation_results.csv'}")

    # =========================================================================
    # STEP 5: MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING")
    print("=" * 70)

    models = {}

    # Persistence Baseline
    print("\n" + "-" * 50)
    print("Training Persistence (Naive) Baseline...")
    print("-" * 50)
    persistence_model = PersistenceModel(lag=1)
    persistence_model.fit(X_train, y_train)
    models['Persistence'] = persistence_model

    # Get persistence predictions for skill scores
    persistence_predictions = X_test['demand_lag_1h'].values if 'demand_lag_1h' in X_test.columns else y_test

    # SARIMA Baseline (Optional - can be slow)
    if not skip_sarima:
        print("\n" + "-" * 50)
        print("Training SARIMA Baseline...")
        print("-" * 50)
        try:
            sarima_model = SARIMAModel(order=(2, 1, 2), seasonal_order=(1, 0, 1, 24))
            sarima_model.fit(y_train, maxiter=50)
            models['SARIMA'] = sarima_model
        except Exception as e:
            print(f"SARIMA training failed: {e}")

    # Random Forest
    print("\n" + "-" * 50)
    print("Training Random Forest...")
    print("-" * 50)
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model

    # XGBoost
    print("\n" + "-" * 50)
    print("Training XGBoost...")
    print("-" * 50)
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb_model

    # LightGBM
    print("\n" + "-" * 50)
    print("Training LightGBM...")
    print("-" * 50)
    try:
        lgb_model = LightGBMModel()
        lgb_model.fit(X_train, y_train)
        models['LightGBM'] = lgb_model
    except Exception as e:
        print(f"LightGBM training failed: {e}")

    # Quantile Regression for Prediction Intervals
    print("\n" + "-" * 50)
    print("Training Quantile Regression for Prediction Intervals...")
    print("-" * 50)
    qr_model = QuantileRegressionModel(quantiles=[0.05, 0.50, 0.95])
    qr_model.fit(X_train, y_train)
    models['Quantile Regression'] = qr_model

    # Save models
    print("\n" + "-" * 50)
    print("Saving models...")
    print("-" * 50)
    for name, model in models.items():
        if name not in ['SARIMA', 'Persistence']:
            model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_model.pkl"
            model.save(model_path)
            print(f"  Saved: {name}")

    results['models'] = models

    # =========================================================================
    # STEP 6: MODEL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 70)

    # Basic evaluation
    eval_models = {k: v for k, v in models.items() if k not in ['SARIMA', 'Quantile Regression']}

    # For SARIMA, we need special handling
    if 'SARIMA' in models:
        print("\nEvaluating SARIMA (separate due to different prediction method)...")
        try:
            sarima_pred = models['SARIMA'].predict(len(y_test))
            sarima_metrics = calculate_metrics(y_test, sarima_pred, 'SARIMA')
            print(f"  SARIMA RMSE: {sarima_metrics['RMSE']:.2f}")
        except Exception as e:
            print(f"  SARIMA prediction failed: {e}")

    evaluation_report = generate_evaluation_report(
        eval_models, X_test, y_test, test_df,
        save_path=RESULTS_DIR
    )
    results['evaluation'] = evaluation_report

    # =========================================================================
    # STEP 7: SKILL SCORES
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: SKILL SCORES (vs Persistence Baseline)")
    print("=" * 70)

    skill_scores = calculate_all_skill_scores(eval_models, X_test, y_test, persistence_predictions)
    print("\nSkill Scores (higher is better, >0 means better than persistence):")
    print(skill_scores.to_string(index=False))

    skill_scores.to_csv(RESULTS_DIR / 'skill_scores.csv', index=False)
    results['skill_scores'] = skill_scores

    # =========================================================================
    # STEP 8: DIEBOLD-MARIANO TEST
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: DIEBOLD-MARIANO STATISTICAL TEST")
    print("=" * 70)

    dm_results = compare_models_dm(eval_models, X_test, y_test, reference_model='XGBoost')
    print("\nDiebold-Mariano Test Results:")
    print(dm_results.to_string(index=False))

    dm_results.to_csv(RESULTS_DIR / 'diebold_mariano_test.csv', index=False)
    results['dm_test'] = dm_results

    # =========================================================================
    # STEP 9: PREDICTION INTERVALS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: PREDICTION INTERVALS (90% Confidence)")
    print("=" * 70)

    pi_predictions = qr_model.predict_intervals(X_test)
    pi_metrics = evaluate_prediction_intervals(
        y_test,
        pi_predictions['lower'],
        pi_predictions['upper'],
        confidence=0.90
    )

    print(f"\nPrediction Interval Evaluation:")
    print(f"  Coverage: {pi_metrics['coverage']:.1%} (expected: 90%)")
    print(f"  Coverage Error: {pi_metrics['coverage_error']:+.1%}")
    print(f"  Average Width: {pi_metrics['avg_interval_width']:.2f} MW")
    print(f"  Average Width %: {pi_metrics['avg_interval_width_pct']:.1f}%")
    print(f"  Interval Score: {pi_metrics['interval_score']:.2f}")

    # Save prediction intervals
    pi_df = pd.DataFrame({
        'actual': y_test,
        'lower': pi_predictions['lower'],
        'median': pi_predictions['median'],
        'upper': pi_predictions['upper']
    })
    pi_df.to_csv(RESULTS_DIR / 'prediction_intervals.csv', index=False)
    results['prediction_intervals'] = pi_metrics

    # =========================================================================
    # STEP 10: ABLATION STUDY (Optional)
    # =========================================================================
    if not skip_ablation:
        print("\n" + "=" * 70)
        print("STEP 10: ABLATION STUDY")
        print("=" * 70)

        ablation_results = generate_ablation_report(
            X_train, y_train, X_test, y_test,
            save_dir=RESULTS_DIR
        )
        results['ablation'] = ablation_results

    # =========================================================================
    # STEP 11: SHAP EXPLAINABILITY (Optional)
    # =========================================================================
    if not skip_shap:
        print("\n" + "=" * 70)
        print("STEP 11: SHAP EXPLAINABILITY ANALYSIS")
        print("=" * 70)

        try:
            shap_results = generate_shap_report(
                xgb_model, X_test, feature_cols,
                save_dir=PLOTS_DIR / 'shap'
            )
            results['shap'] = shap_results
        except Exception as e:
            print(f"SHAP analysis failed: {e}")

    # =========================================================================
    # STEP 12: VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 12: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Model comparison plot
    if 'overall' in evaluation_report:
        plot_model_comparison(
            evaluation_report['overall'],
            save_path=PLOTS_DIR / 'model_comparison.png'
        )

    # Best model predictions
    best_model_name = evaluation_report['overall'].iloc[0]['model']
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)

    plot_predictions(
        y_test, y_pred,
        dates=test_df.index,
        model_name=best_model_name,
        save_path=PLOTS_DIR / f'{best_model_name.lower().replace(" ", "_")}_predictions.png'
    )

    # Feature importance
    if hasattr(best_model, 'get_feature_importance'):
        importance = best_model.get_feature_importance()
        plot_feature_importance(
            importance,
            title=f'{best_model_name} Feature Importance',
            save_path=PLOTS_DIR / f'{best_model_name.lower().replace(" ", "_")}_feature_importance.png'
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("DATA SUMMARY")
    print("-" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {len(feature_cols)}")

    print("\n" + "-" * 50)
    print("MODEL PERFORMANCE")
    print("-" * 50)
    print(evaluation_report['overall'].to_string(index=False))

    print("\n" + "-" * 50)
    print("BEST MODEL")
    print("-" * 50)
    best = evaluation_report['overall'].iloc[0]
    print(f"Model: {best['model']}")
    print(f"RMSE: {best['RMSE']:.2f} MW")
    print(f"MAPE: {best['MAPE']:.2f}%")
    print(f"R²: {best['R2']:.4f}")

    if 'skill_scores' in results:
        best_skill = skill_scores[skill_scores['model'] == best['model']]['skill_score'].values
        if len(best_skill) > 0:
            print(f"Skill Score vs Persistence: {best_skill[0]:.4f}")

    # Save final summary
    summary_path = RESULTS_DIR / 'enhanced_summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IRELAND ELECTRICITY DEMAND FORECASTING - ENHANCED SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SUMMARY:\n")
        f.write(f"  Training samples: {len(X_train)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Features: {len(feature_cols)}\n\n")

        f.write("MODEL PERFORMANCE:\n")
        f.write(evaluation_report['overall'].to_string(index=False))
        f.write("\n\n")

        f.write("SKILL SCORES:\n")
        f.write(skill_scores.to_string(index=False))
        f.write("\n\n")

        f.write("PREDICTION INTERVALS:\n")
        f.write(f"  Coverage: {pi_metrics['coverage']:.1%}\n")
        f.write(f"  Average Width: {pi_metrics['avg_interval_width']:.2f} MW\n")

    print(f"\nSummary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("ENHANCED PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Ireland Electricity Demand Forecasting')
    parser.add_argument('--force-download', action='store_true', help='Force re-download of OPSD data')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--skip-sarima', action='store_true', help='Skip SARIMA (faster)')
    parser.add_argument('--skip-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--skip-ablation', action='store_true', help='Skip ablation study')

    args = parser.parse_args()

    results = run_enhanced_pipeline(
        force_download=args.force_download,
        skip_eda=args.skip_eda,
        skip_shap=args.skip_shap,
        skip_sarima=args.skip_sarima,
        skip_cv=args.skip_cv,
        skip_ablation=args.skip_ablation
    )
