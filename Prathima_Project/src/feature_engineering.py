"""
Feature Engineering Module for Electricity Demand Forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import LAG_FEATURES, ROLLING_WINDOWS


def create_temporal_features(df):
    """
    Create calendar/temporal features from datetime index.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index

    Returns:
    --------
    pd.DataFrame : Dataframe with temporal features added
    """
    df = df.copy()

    # Basic temporal features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter

    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)

    # Cyclical encoding for hour and month (sine/cosine transformation)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print(f"Created {12 + 6} temporal features")

    return df


def create_lag_features(df, target_col='demand', lags=LAG_FEATURES):
    """
    Create lagged features for the target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to create lags for
    lags : list
        List of lag periods (in hours)

    Returns:
    --------
    pd.DataFrame : Dataframe with lag features
    """
    df = df.copy()

    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)

    print(f"Created {len(lags)} lag features: {lags}")

    return df


def create_rolling_features(df, target_col='demand', windows=ROLLING_WINDOWS):
    """
    Create rolling statistics features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to create rolling stats for
    windows : list
        List of rolling window sizes (in hours)

    Returns:
    --------
    pd.DataFrame : Dataframe with rolling features
    """
    df = df.copy()

    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}h'] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{target_col}_rolling_std_{window}h'] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        )
        # Rolling min
        df[f'{target_col}_rolling_min_{window}h'] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).min()
        )
        # Rolling max
        df[f'{target_col}_rolling_max_{window}h'] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).max()
        )

    print(f"Created {len(windows) * 4} rolling features for windows: {windows}")

    return df


def create_difference_features(df, target_col='demand'):
    """
    Create difference/change features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to create differences for

    Returns:
    --------
    pd.DataFrame : Dataframe with difference features
    """
    df = df.copy()

    # Hour-over-hour change
    df[f'{target_col}_diff_1h'] = df[target_col].diff(1)

    # Day-over-day change (same hour)
    df[f'{target_col}_diff_24h'] = df[target_col].diff(24)

    # Week-over-week change (same hour, same day)
    df[f'{target_col}_diff_168h'] = df[target_col].diff(168)

    # Percentage changes
    df[f'{target_col}_pct_change_1h'] = df[target_col].pct_change(1)
    df[f'{target_col}_pct_change_24h'] = df[target_col].pct_change(24)

    print("Created 5 difference features")

    return df


def create_holiday_features(df, country='IE'):
    """
    Create holiday features for Ireland.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    country : str
        Country code

    Returns:
    --------
    pd.DataFrame : Dataframe with holiday features
    """
    df = df.copy()

    try:
        import holidays
        ie_holidays = holidays.Ireland(years=df.index.year.unique().tolist())
        holiday_dates = pd.to_datetime(list(ie_holidays.keys())).date
        df['is_holiday'] = pd.Series(df.index.date, index=df.index).isin(holiday_dates).astype(int)
        print("Created holiday feature using holidays library")
    except (ImportError, Exception) as e:
        # Manual Irish bank holidays (approximate)
        df['is_holiday'] = 0
        # Christmas period
        df.loc[(df.index.month == 12) & (df.index.day >= 24), 'is_holiday'] = 1
        df.loc[(df.index.month == 1) & (df.index.day <= 2), 'is_holiday'] = 1
        print("Created approximate holiday feature (holidays library not installed)")

    # Day before/after holiday effect
    df['is_near_holiday'] = (
        (df['is_holiday'].shift(1).fillna(0).astype(int) == 1) |
        (df['is_holiday'].shift(-1).fillna(0).astype(int) == 1)
    ).astype(int)

    return df


def create_interaction_features(df):
    """
    Create interaction features between variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame : Dataframe with interaction features
    """
    df = df.copy()

    # Hour and weekend interaction
    df['hour_weekend'] = df['hour'] * df['is_weekend']

    # Hour and month interaction (seasonal hourly patterns)
    df['hour_month'] = df['hour'] * df['month']

    # Peak hours on weekdays
    df['peak_weekday'] = (
        ((df['is_peak_morning'] == 1) | (df['is_peak_evening'] == 1)) &
        (df['is_weekend'] == 0)
    ).astype(int)

    print("Created 3 interaction features")

    return df


def engineer_features(df, include_weather=True):
    """
    Complete feature engineering pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe with demand column
    include_weather : bool
        Whether to include weather features if available

    Returns:
    --------
    pd.DataFrame : Dataframe with all engineered features
    """
    print("=" * 50)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    initial_cols = len(df.columns)

    # Temporal features
    print("\n1. Creating temporal features...")
    df = create_temporal_features(df)

    # Lag features
    print("\n2. Creating lag features...")
    df = create_lag_features(df)

    # Rolling features
    print("\n3. Creating rolling features...")
    df = create_rolling_features(df)

    # Difference features
    print("\n4. Creating difference features...")
    df = create_difference_features(df)

    # Holiday features
    print("\n5. Creating holiday features...")
    df = create_holiday_features(df)

    # Interaction features
    print("\n6. Creating interaction features...")
    df = create_interaction_features(df)

    # Handle any remaining NaN from feature creation
    print("\n7. Handling NaN values from feature creation...")
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with NaN values")

    final_cols = len(df.columns)
    print(f"\nFeature engineering complete!")
    print(f"Total features created: {final_cols - initial_cols}")
    print(f"Final dataframe shape: {df.shape}")

    return df


def get_feature_columns(df, exclude_target=True):
    """
    Get list of feature column names.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    exclude_target : bool
        Whether to exclude target column

    Returns:
    --------
    list : List of feature column names
    """
    exclude_cols = ['demand', 'demand_scaled'] if exclude_target else []
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.data_loader import load_and_prepare_data

    df = load_and_prepare_data()
    df_features = engineer_features(df)
    print("\nFeature columns:")
    print(df_features.columns.tolist())
