"""
Explainability Module using SHAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config.config import PLOTS_DIR


def compute_shap_values(model, X, feature_names=None, sample_size=1000):
    """
    Compute SHAP values for model predictions.

    Parameters:
    -----------
    model : trained model
        XGBoost, Random Forest, or similar tree-based model
    X : pd.DataFrame or np.ndarray
        Feature data
    feature_names : list, optional
        Names of features
    sample_size : int
        Number of samples to use for SHAP calculation

    Returns:
    --------
    tuple : (shap_values, explainer)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library required. Install with: pip install shap")

    print("Computing SHAP values...")

    # Sample data if too large
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        if hasattr(X, 'iloc'):
            X_sample = X.iloc[indices]
        else:
            X_sample = X[indices]
    else:
        X_sample = X

    # Get the underlying model
    if hasattr(model, 'model'):
        underlying_model = model.model
    else:
        underlying_model = model

    # Create appropriate explainer based on model type
    model_type = type(underlying_model).__name__

    if 'XGB' in model_type or 'RandomForest' in model_type or 'Gradient' in model_type:
        explainer = shap.TreeExplainer(underlying_model)
    else:
        # Use KernelExplainer for other models
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(underlying_model.predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values

    print(f"SHAP values computed for {len(X_sample)} samples")

    return shap_values, explainer, X_sample


def plot_shap_summary(shap_values, X, feature_names=None, max_display=20, save_path=None):
    """
    Create SHAP summary plot.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame or np.ndarray
        Feature data
    feature_names : list, optional
        Names of features
    max_display : int
        Maximum number of features to display
    save_path : Path, optional
        Path to save plot
    """
    import shap

    plt.figure(figsize=(12, 10))

    if feature_names is not None and hasattr(X, 'values'):
        X_display = pd.DataFrame(X.values, columns=feature_names)
    else:
        X_display = X

    shap.summary_plot(
        shap_values,
        X_display,
        max_display=max_display,
        show=False
    )

    plt.title('SHAP Feature Importance Summary', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {save_path}")

    plt.show()


def plot_shap_bar(shap_values, feature_names, max_display=20, save_path=None):
    """
    Create SHAP bar plot (mean absolute SHAP values).

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        Names of features
    max_display : int
        Maximum number of features to display
    save_path : Path, optional
        Path to save plot
    """
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=True).tail(max_display)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance (Mean Absolute SHAP Values)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to {save_path}")

    plt.show()

    return importance_df


def plot_shap_dependence(shap_values, X, feature_name, interaction_feature=None, save_path=None):
    """
    Create SHAP dependence plot for a specific feature.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame
        Feature data
    feature_name : str
        Feature to plot
    interaction_feature : str, optional
        Feature for coloring
    save_path : Path, optional
        Path to save plot
    """
    import shap

    plt.figure(figsize=(10, 6))

    if hasattr(X, 'columns'):
        feature_idx = list(X.columns).index(feature_name)
    else:
        feature_idx = feature_name

    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        interaction_index=interaction_feature,
        show=False
    )

    plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dependence plot saved to {save_path}")

    plt.show()


def plot_local_explanation(shap_values, X, index, feature_names=None, save_path=None):
    """
    Create force plot for a single prediction explanation.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame or np.ndarray
        Feature data
    index : int
        Index of sample to explain
    feature_names : list, optional
        Names of features
    save_path : Path, optional
        Path to save plot
    """
    import shap

    # Get expected value
    expected_value = shap_values.mean()

    plt.figure(figsize=(14, 4))

    if feature_names is not None:
        feature_display = feature_names
    elif hasattr(X, 'columns'):
        feature_display = X.columns.tolist()
    else:
        feature_display = [f'Feature {i}' for i in range(X.shape[1])]

    # Create waterfall plot instead of force plot for better static visualization
    shap_val = shap_values[index]
    if hasattr(X, 'iloc'):
        feature_val = X.iloc[index].values
    else:
        feature_val = X[index]

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_val))[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#ff0051' if v > 0 else '#008bfb' for v in shap_val[sorted_idx]]

    y_pos = range(len(sorted_idx))
    ax.barh(y_pos, shap_val[sorted_idx], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{feature_display[i]} = {feature_val[i]:.2f}' for i in sorted_idx])
    ax.set_xlabel('SHAP Value (impact on prediction)')
    ax.set_title(f'Local Explanation for Sample {index}')
    ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Local explanation saved to {save_path}")

    plt.show()


def analyze_temporal_shap(shap_values, X, feature_names):
    """
    Analyze SHAP values for temporal features.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame
        Feature data
    feature_names : list
        Names of features

    Returns:
    --------
    dict : Analysis results for temporal features
    """
    temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                        'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    available_temporal = [f for f in temporal_features if f in feature_names]

    results = {}

    for feature in available_temporal:
        idx = feature_names.index(feature)
        shap_col = shap_values[:, idx]

        results[feature] = {
            'mean_abs_shap': np.abs(shap_col).mean(),
            'mean_shap': shap_col.mean(),
            'std_shap': shap_col.std(),
            'positive_impact_pct': (shap_col > 0).mean() * 100
        }

    # Sort by importance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_abs_shap'], reverse=True)

    print("\n" + "=" * 50)
    print("TEMPORAL FEATURE IMPORTANCE (SHAP)")
    print("=" * 50)

    for feature, metrics in sorted_results:
        print(f"\n{feature}:")
        print(f"  Mean |SHAP|: {metrics['mean_abs_shap']:.4f}")
        print(f"  Positive impact: {metrics['positive_impact_pct']:.1f}% of samples")

    return dict(sorted_results)


def generate_shap_report(model, X_test, feature_names, save_dir=None):
    """
    Generate comprehensive SHAP explainability report.

    Parameters:
    -----------
    model : trained model
        Model to explain
    X_test : pd.DataFrame
        Test features
    feature_names : list
        Names of features
    save_dir : Path, optional
        Directory to save plots

    Returns:
    --------
    dict : Complete SHAP analysis results
    """
    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Compute SHAP values
    shap_values, explainer, X_sample = compute_shap_values(
        model, X_test, feature_names
    )

    results = {
        'shap_values': shap_values,
        'X_sample': X_sample
    }

    # Global importance
    print("\n1. Computing global feature importance...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    results['importance'] = importance_df
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Summary plot
    print("\n2. Creating summary plot...")
    plot_shap_summary(
        shap_values, X_sample, feature_names,
        save_path=save_dir / 'shap_summary.png' if save_dir else None
    )

    # Bar plot
    print("\n3. Creating bar plot...")
    plot_shap_bar(
        shap_values, feature_names,
        save_path=save_dir / 'shap_bar.png' if save_dir else None
    )

    # Temporal analysis
    print("\n4. Analyzing temporal features...")
    results['temporal_analysis'] = analyze_temporal_shap(shap_values, X_sample, feature_names)

    # Dependence plots for top features
    print("\n5. Creating dependence plots for top features...")
    top_features = importance_df.head(3)['feature'].tolist()

    for feature in top_features:
        if feature in feature_names:
            plot_shap_dependence(
                shap_values, X_sample, feature,
                save_path=save_dir / f'shap_dependence_{feature}.png' if save_dir else None
            )

    # Local explanations
    print("\n6. Creating sample local explanations...")
    for i, idx in enumerate([0, len(X_sample)//2, len(X_sample)-1]):
        plot_local_explanation(
            shap_values, X_sample, idx, feature_names,
            save_path=save_dir / f'shap_local_{i}.png' if save_dir else None
        )

    print("\nSHAP analysis complete!")

    if save_dir:
        importance_df.to_csv(save_dir / 'feature_importance_shap.csv', index=False)
        print(f"Results saved to {save_dir}")

    return results
