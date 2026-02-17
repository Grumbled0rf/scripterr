"""
Ablation Study Module
Analyzes the contribution of different feature groups to model performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from config.config import PLOTS_DIR, RESULTS_DIR


# Feature group definitions
FEATURE_GROUPS = {
    'temporal': [
        'hour', 'day_of_week', 'day_of_month', 'month', 'year',
        'week_of_year', 'quarter', 'is_weekend', 'is_night',
        'is_peak_morning', 'is_peak_evening', 'hour_sin', 'hour_cos',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
    ],
    'lag': [
        'demand_lag_1h', 'demand_lag_2h', 'demand_lag_3h',
        'demand_lag_24h', 'demand_lag_48h', 'demand_lag_168h'
    ],
    'rolling': [
        'demand_rolling_mean_6h', 'demand_rolling_std_6h',
        'demand_rolling_min_6h', 'demand_rolling_max_6h',
        'demand_rolling_mean_12h', 'demand_rolling_std_12h',
        'demand_rolling_min_12h', 'demand_rolling_max_12h',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',
        'demand_rolling_min_24h', 'demand_rolling_max_24h',
        'demand_rolling_mean_48h', 'demand_rolling_std_48h',
        'demand_rolling_min_48h', 'demand_rolling_max_48h',
        'demand_rolling_mean_168h', 'demand_rolling_std_168h',
        'demand_rolling_min_168h', 'demand_rolling_max_168h'
    ],
    'difference': [
        'demand_diff_1h', 'demand_diff_24h', 'demand_diff_168h',
        'demand_pct_change_1h', 'demand_pct_change_24h'
    ],
    'holiday': [
        'is_holiday', 'is_near_holiday'
    ],
    'interaction': [
        'hour_weekend', 'hour_month', 'peak_weekday'
    ]
}


def get_feature_group_columns(df, group_name):
    """Get columns belonging to a feature group that exist in dataframe."""
    if group_name not in FEATURE_GROUPS:
        return []

    available = [col for col in FEATURE_GROUPS[group_name] if col in df.columns]
    return available


def run_ablation_study(X_train, y_train, X_test, y_test, model_class, model_params=None):
    """
    Run ablation study by removing one feature group at a time.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    model_class : Model class to use
    model_params : Optional model parameters

    Returns:
    --------
    pd.DataFrame : Ablation results
    """
    from src.evaluation import calculate_metrics

    if model_params is None:
        model_params = {}

    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    results = []

    # Baseline: all features
    print("\n[Baseline] Training with all features...")
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_metrics = calculate_metrics(y_test, y_pred, "All Features")
    baseline_metrics['removed_group'] = 'None (Baseline)'
    baseline_metrics['n_features'] = X_train.shape[1]
    results.append(baseline_metrics)

    print(f"  Features: {X_train.shape[1]}, RMSE: {baseline_metrics['RMSE']:.2f}")

    # Remove each feature group
    for group_name, group_features in FEATURE_GROUPS.items():
        print(f"\n[Ablation] Removing {group_name} features...")

        # Get features to keep
        features_to_remove = [f for f in group_features if f in X_train.columns]
        features_to_keep = [f for f in X_train.columns if f not in features_to_remove]

        if len(features_to_remove) == 0:
            print(f"  No {group_name} features found, skipping...")
            continue

        if len(features_to_keep) == 0:
            print(f"  Cannot remove all features, skipping...")
            continue

        # Train model without this feature group
        X_train_ablated = X_train[features_to_keep]
        X_test_ablated = X_test[features_to_keep]

        model = model_class(**model_params)
        model.fit(X_train_ablated, y_train)
        y_pred = model.predict(X_test_ablated)

        metrics = calculate_metrics(y_test, y_pred, f"Without {group_name}")
        metrics['removed_group'] = group_name
        metrics['n_features'] = len(features_to_keep)
        metrics['n_removed'] = len(features_to_remove)
        metrics['rmse_change'] = metrics['RMSE'] - baseline_metrics['RMSE']
        metrics['rmse_change_pct'] = (metrics['RMSE'] - baseline_metrics['RMSE']) / baseline_metrics['RMSE'] * 100

        results.append(metrics)

        print(f"  Removed: {len(features_to_remove)} features")
        print(f"  Remaining: {len(features_to_keep)} features")
        print(f"  RMSE: {metrics['RMSE']:.2f} (Δ: {metrics['rmse_change']:+.2f}, {metrics['rmse_change_pct']:+.1f}%)")

    results_df = pd.DataFrame(results)

    # Sort by impact (RMSE change)
    results_df = results_df.sort_values('RMSE', ascending=True)

    return results_df


def run_feature_addition_study(X_train, y_train, X_test, y_test, model_class, model_params=None):
    """
    Run feature addition study by adding one feature group at a time.
    Shows incremental value of each feature group.

    Returns:
    --------
    pd.DataFrame : Feature addition results
    """
    from src.evaluation import calculate_metrics

    if model_params is None:
        model_params = {}

    print("\n" + "=" * 60)
    print("FEATURE ADDITION STUDY")
    print("=" * 60)

    results = []
    current_features = []

    # Order of feature groups to add (most to least important typically)
    group_order = ['lag', 'rolling', 'temporal', 'difference', 'interaction', 'holiday']

    for group_name in group_order:
        group_features = [f for f in FEATURE_GROUPS.get(group_name, []) if f in X_train.columns]

        if len(group_features) == 0:
            continue

        current_features.extend(group_features)

        print(f"\n[Addition] Adding {group_name} features...")
        print(f"  Total features: {len(current_features)}")

        X_train_subset = X_train[current_features]
        X_test_subset = X_test[current_features]

        model = model_class(**model_params)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_test_subset)

        metrics = calculate_metrics(y_test, y_pred, f"+ {group_name}")
        metrics['added_group'] = group_name
        metrics['n_features'] = len(current_features)
        metrics['cumulative_groups'] = ', '.join(group_order[:group_order.index(group_name)+1])

        results.append(metrics)

        print(f"  RMSE: {metrics['RMSE']:.2f}")

    results_df = pd.DataFrame(results)

    return results_df


def plot_ablation_results(ablation_df, save_path=None):
    """
    Create visualization of ablation study results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out baseline for comparison plot
    comparison_df = ablation_df[ablation_df['removed_group'] != 'None (Baseline)'].copy()

    if len(comparison_df) == 0:
        print("No ablation results to plot")
        return

    # Sort by RMSE change
    comparison_df = comparison_df.sort_values('rmse_change', ascending=False)

    # Plot 1: RMSE change when removing each group
    colors = ['red' if x > 0 else 'green' for x in comparison_df['rmse_change']]
    axes[0].barh(comparison_df['removed_group'], comparison_df['rmse_change'], color=colors)
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('RMSE Change (MW)')
    axes[0].set_title('Impact of Removing Feature Groups\n(Positive = Worse Performance)')

    # Plot 2: Absolute RMSE comparison
    all_results = ablation_df.sort_values('RMSE')
    colors = ['green' if x == 'None (Baseline)' else 'steelblue' for x in all_results['removed_group']]
    axes[1].barh(all_results['removed_group'], all_results['RMSE'], color=colors)
    axes[1].set_xlabel('RMSE (MW)')
    axes[1].set_title('Model Performance by Feature Configuration')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation plot saved to {save_path}")

    plt.close(fig)  # Close instead of show for non-interactive mode

    return fig


def generate_ablation_report(X_train, y_train, X_test, y_test, save_dir=None):
    """
    Generate comprehensive ablation study report.
    """
    from xgboost import XGBRegressor

    print("\n" + "=" * 60)
    print("COMPREHENSIVE ABLATION ANALYSIS")
    print("=" * 60)

    # XGBoost parameters
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }

    results = {}

    # Run ablation study
    print("\n1. Feature Removal Study")
    ablation_results = run_ablation_study(
        X_train, y_train, X_test, y_test,
        XGBRegressor, xgb_params
    )
    results['ablation'] = ablation_results

    # Run feature addition study
    print("\n2. Feature Addition Study")
    addition_results = run_feature_addition_study(
        X_train, y_train, X_test, y_test,
        XGBRegressor, xgb_params
    )
    results['addition'] = addition_results

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    print("\nFeature Group Importance (by RMSE impact when removed):")
    if 'rmse_change' in ablation_results.columns:
        importance_order = ablation_results[ablation_results['removed_group'] != 'None (Baseline)']
        importance_order = importance_order.sort_values('rmse_change', ascending=False)

        for _, row in importance_order.iterrows():
            impact = "Critical" if row['rmse_change'] > 10 else "Important" if row['rmse_change'] > 1 else "Minor"
            print(f"  {row['removed_group']:15} | RMSE Δ: {row['rmse_change']:+7.2f} MW | {impact}")

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ablation_results.to_csv(save_dir / 'ablation_results.csv', index=False)
        addition_results.to_csv(save_dir / 'feature_addition_results.csv', index=False)

        # Plot
        plot_ablation_results(ablation_results, save_dir / 'ablation_plot.png')

        print(f"\nResults saved to {save_dir}")

    return results


if __name__ == "__main__":
    # Test ablation study
    print("Ablation study module loaded successfully")
