"""
Visualization Module for Electricity Demand Forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import PLOTS_DIR

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_demand_overview(df, save_path=None):
    """
    Create overview plots of demand data.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with demand column and datetime index
    save_path : Path, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Time series
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['demand'], linewidth=0.5, alpha=0.7)
    ax1.set_title('Electricity Demand Over Time', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Demand (MW)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Distribution
    ax2 = axes[0, 1]
    ax2.hist(df['demand'], bins=50, edgecolor='white', alpha=0.7)
    ax2.axvline(df['demand'].mean(), color='red', linestyle='--', label=f'Mean: {df["demand"].mean():.0f}')
    ax2.axvline(df['demand'].median(), color='orange', linestyle='--', label=f'Median: {df["demand"].median():.0f}')
    ax2.set_title('Demand Distribution', fontsize=12)
    ax2.set_xlabel('Demand (MW)')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Daily pattern
    ax3 = axes[1, 0]
    hourly_mean = df.groupby(df.index.hour)['demand'].mean()
    hourly_std = df.groupby(df.index.hour)['demand'].std()
    ax3.fill_between(hourly_mean.index, hourly_mean - hourly_std, hourly_mean + hourly_std, alpha=0.3)
    ax3.plot(hourly_mean.index, hourly_mean, linewidth=2, marker='o')
    ax3.set_title('Average Daily Demand Pattern', fontsize=12)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Demand (MW)')
    ax3.set_xticks(range(0, 24, 2))

    # Weekly pattern
    ax4 = axes[1, 1]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_mean = df.groupby(df.index.dayofweek)['demand'].mean()
    daily_std = df.groupby(df.index.dayofweek)['demand'].std()
    ax4.bar(range(7), daily_mean, yerr=daily_std, capsize=3, alpha=0.7)
    ax4.set_title('Average Weekly Demand Pattern', fontsize=12)
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Demand (MW)')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(days)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overview plot saved to {save_path}")

    plt.show()


def plot_seasonal_patterns(df, save_path=None):
    """
    Create seasonal pattern analysis plots.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with demand column and datetime index
    save_path : Path, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Monthly pattern
    ax1 = axes[0, 0]
    monthly_mean = df.groupby(df.index.month)['demand'].mean()
    monthly_std = df.groupby(df.index.month)['demand'].std()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1.bar(range(1, 13), monthly_mean, yerr=monthly_std, capsize=3, alpha=0.7)
    ax1.set_title('Monthly Average Demand', fontsize=12)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Demand (MW)')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(months, rotation=45)

    # Heatmap: Hour vs Day of Week
    ax2 = axes[0, 1]
    pivot = df.pivot_table(values='demand', index=df.index.hour, columns=df.index.dayofweek, aggfunc='mean')
    pivot.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.heatmap(pivot, ax=ax2, cmap='YlOrRd', annot=False, fmt='.0f')
    ax2.set_title('Demand Heatmap: Hour vs Day', fontsize=12)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Hour of Day')

    # Heatmap: Hour vs Month
    ax3 = axes[1, 0]
    pivot2 = df.pivot_table(values='demand', index=df.index.hour, columns=df.index.month, aggfunc='mean')
    pivot2.columns = months
    sns.heatmap(pivot2, ax=ax3, cmap='YlOrRd', annot=False, fmt='.0f')
    ax3.set_title('Demand Heatmap: Hour vs Month', fontsize=12)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Hour of Day')

    # Year over year comparison
    ax4 = axes[1, 1]
    for year in df.index.year.unique():
        yearly_data = df[df.index.year == year]
        daily_mean = yearly_data.groupby(yearly_data.index.dayofyear)['demand'].mean()
        ax4.plot(daily_mean.index, daily_mean, label=str(year), alpha=0.7)
    ax4.set_title('Year-over-Year Daily Demand', fontsize=12)
    ax4.set_xlabel('Day of Year')
    ax4.set_ylabel('Demand (MW)')
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Seasonal patterns plot saved to {save_path}")

    plt.show()


def plot_predictions(y_true, y_pred, dates=None, model_name="Model", n_points=500, save_path=None):
    """
    Plot actual vs predicted values.

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    dates : array-like, optional
        Datetime index
    model_name : str
        Name of model for title
    n_points : int
        Number of points to plot
    save_path : Path, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Limit points for visualization
    if len(y_true) > n_points:
        indices = np.linspace(0, len(y_true)-1, n_points, dtype=int)
        y_true_plot = np.array(y_true)[indices]
        y_pred_plot = np.array(y_pred)[indices]
        dates_plot = dates[indices] if dates is not None else range(n_points)
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        dates_plot = dates if dates is not None else range(len(y_true))

    # Time series comparison
    ax1 = axes[0, 0]
    ax1.plot(dates_plot, y_true_plot, label='Actual', alpha=0.7, linewidth=1)
    ax1.plot(dates_plot, y_pred_plot, label='Predicted', alpha=0.7, linewidth=1)
    ax1.set_title(f'{model_name}: Actual vs Predicted', fontsize=12)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Demand (MW)')
    ax1.legend()
    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax2.set_title(f'{model_name}: Prediction Scatter Plot', fontsize=12)
    ax2.set_xlabel('Actual Demand (MW)')
    ax2.set_ylabel('Predicted Demand (MW)')
    ax2.legend()

    # Residuals
    residuals = np.array(y_true) - np.array(y_pred)

    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, edgecolor='white', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_title(f'{model_name}: Residual Distribution', fontsize=12)
    ax3.set_xlabel('Residual (Actual - Predicted)')
    ax3.set_ylabel('Frequency')

    # Residuals over time
    ax4 = axes[1, 1]
    ax4.scatter(dates_plot, residuals[indices] if len(y_true) > n_points else residuals, alpha=0.3, s=10)
    ax4.axhline(0, color='red', linestyle='--')
    ax4.set_title(f'{model_name}: Residuals Over Time', fontsize=12)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Residual (MW)')
    if dates is not None:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")

    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot comparison of model metrics.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model comparison metrics
    save_path : Path, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    colors = sns.color_palette("husl", len(results_df))

    for ax, metric in zip(axes.flatten(), metrics):
        bars = ax.bar(results_df['model'], results_df[metric], color=colors)
        ax.set_title(f'{metric} Comparison', fontsize=12)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, val in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")

    plt.show()


def plot_feature_importance(importance_df, top_n=20, title="Feature Importance", save_path=None):
    """
    Plot feature importance.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    title : str
        Plot title
    save_path : Path, optional
        Path to save plot
    """
    plt.figure(figsize=(10, 8))

    top_features = importance_df.head(top_n).sort_values('importance', ascending=True)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    plt.barh(top_features['feature'], top_features['importance'], color=colors)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_hourly_performance(hourly_df, metric='MAPE', save_path=None):
    """
    Plot model performance by hour of day.

    Parameters:
    -----------
    hourly_df : pd.DataFrame
        DataFrame with hourly metrics
    metric : str
        Metric to plot
    save_path : Path, optional
        Path to save plot
    """
    plt.figure(figsize=(12, 6))

    plt.bar(hourly_df['hour'], hourly_df[metric], color='steelblue', alpha=0.7)
    plt.axhline(hourly_df[metric].mean(), color='red', linestyle='--',
                label=f'Mean {metric}: {hourly_df[metric].mean():.2f}')

    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'Model Performance by Hour of Day ({metric})', fontsize=14)
    plt.xticks(range(24))
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hourly performance plot saved to {save_path}")

    plt.show()


def create_eda_report(df, save_dir=None):
    """
    Create comprehensive EDA report with all visualizations.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with demand data
    save_dir : Path, optional
        Directory to save plots
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 60)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Basic statistics
    print("\n1. Basic Statistics:")
    print(df['demand'].describe())

    # Overview plots
    print("\n2. Creating overview plots...")
    plot_demand_overview(
        df,
        save_path=save_dir / 'demand_overview.png' if save_dir else None
    )

    # Seasonal patterns
    print("\n3. Creating seasonal pattern plots...")
    plot_seasonal_patterns(
        df,
        save_path=save_dir / 'seasonal_patterns.png' if save_dir else None
    )

    print("\nEDA report complete!")
