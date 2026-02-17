"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import TRAIN_TEST_SPLIT, VALIDATION_SPLIT, RANDOM_STATE


def handle_missing_values(df, method='interpolate'):
    """
    Handle missing values in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Method to handle missing values ('interpolate', 'ffill', 'drop')

    Returns:
    --------
    pd.DataFrame : Dataframe with handled missing values
    """
    print(f"Missing values before handling:\n{df.isnull().sum()}")

    if method == 'interpolate':
        df = df.interpolate(method='time', limit=24)  # Limit to 24 hours
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'drop':
        df = df.dropna()

    print(f"Missing values after handling:\n{df.isnull().sum()}")

    return df


def remove_outliers(df, column='demand', method='iqr', threshold=3):
    """
    Remove or cap outliers in the data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column to check for outliers
    method : str
        Method to detect outliers ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection

    Returns:
    --------
    pd.DataFrame : Dataframe with handled outliers
    """
    original_len = len(df)

    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Cap outliers instead of removing
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    print(f"Outliers capped. Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    return df


def split_data(df, train_ratio=TRAIN_TEST_SPLIT, val_ratio=VALIDATION_SPLIT):
    """
    Split data into train, validation, and test sets (time-series aware).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    train_ratio : float
        Proportion for training
    val_ratio : float
        Proportion for validation (from remaining data)

    Returns:
    --------
    tuple : (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Data split:")
    print(f"  Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Validation: {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
    print(f"  Test: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, val_df, test_df


def scale_features(train_df, val_df, test_df, feature_columns, scaler_type='standard'):
    """
    Scale features using training data statistics.

    Parameters:
    -----------
    train_df, val_df, test_df : pd.DataFrame
        Data splits
    feature_columns : list
        Columns to scale
    scaler_type : str
        Type of scaler ('standard', 'minmax')

    Returns:
    --------
    tuple : Scaled dataframes and fitted scaler
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # Fit on training data only
    scaler.fit(train_df[feature_columns])

    # Transform all splits
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_columns] = scaler.transform(train_df[feature_columns])
    val_scaled[feature_columns] = scaler.transform(val_df[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_df[feature_columns])

    return train_scaled, val_scaled, test_scaled, scaler


def create_sequences(data, target, sequence_length):
    """
    Create sequences for LSTM model.

    Parameters:
    -----------
    data : np.ndarray
        Feature data
    target : np.ndarray
        Target data
    sequence_length : int
        Length of input sequences

    Returns:
    --------
    tuple : (X, y) sequences
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])

    return np.array(X), np.array(y)


def preprocess_pipeline(df, scale=True):
    """
    Complete preprocessing pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with features
    scale : bool
        Whether to scale features

    Returns:
    --------
    dict : Dictionary containing processed data and scalers
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)

    # Handle missing values
    print("\n1. Handling missing values...")
    df = handle_missing_values(df)

    # Remove outliers
    print("\n2. Handling outliers...")
    df = remove_outliers(df)

    # Split data
    print("\n3. Splitting data...")
    train_df, val_df, test_df = split_data(df)

    result = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'full_data': df
    }

    if scale:
        print("\n4. Scaling features...")
        feature_cols = [col for col in df.columns if col != 'demand']
        if feature_cols:
            train_scaled, val_scaled, test_scaled, scaler = scale_features(
                train_df, val_df, test_df, feature_cols
            )
            result['train_scaled'] = train_scaled
            result['val_scaled'] = val_scaled
            result['test_scaled'] = test_scaled
            result['feature_scaler'] = scaler

        # Scale target separately
        target_scaler = MinMaxScaler()
        result['train']['demand_scaled'] = target_scaler.fit_transform(
            train_df[['demand']]
        )
        result['val']['demand_scaled'] = target_scaler.transform(
            val_df[['demand']]
        )
        result['test']['demand_scaled'] = target_scaler.transform(
            test_df[['demand']]
        )
        result['target_scaler'] = target_scaler

    print("\nPreprocessing complete!")
    return result
