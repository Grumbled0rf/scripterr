"""
Data Loader Module for OPSD Electricity Demand Data
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    DATA_DIR, OPSD_TIME_SERIES_URL, COUNTRY,
    TARGET_COLUMN, DATE_COLUMN
)


def download_opsd_data(force_download=False):
    """
    Download OPSD time series data if not already present.

    Parameters:
    -----------
    force_download : bool
        If True, download even if file exists

    Returns:
    --------
    Path : Path to downloaded file
    """
    file_path = DATA_DIR / "opsd_time_series.csv"

    if file_path.exists() and not force_download:
        print(f"Data file already exists at {file_path}")
        return file_path

    print("Downloading OPSD time series data...")
    print("This may take a few minutes as the file is large (~400MB)...")

    try:
        response = requests.get(OPSD_TIME_SERIES_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)

        print(f"\nData downloaded successfully to {file_path}")
        return file_path

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def load_ireland_data(file_path=None):
    """
    Load and filter OPSD data for Ireland.

    Parameters:
    -----------
    file_path : Path or str, optional
        Path to the OPSD CSV file

    Returns:
    --------
    pd.DataFrame : Ireland electricity demand data
    """
    if file_path is None:
        file_path = DATA_DIR / "opsd_time_series.csv"

    print("Loading OPSD data...")

    # Read only relevant columns to save memory
    # First, read a small sample to identify columns
    sample = pd.read_csv(file_path, nrows=5)

    # Find Ireland-related columns
    ie_columns = [col for col in sample.columns if col.startswith('IE_') or col == DATE_COLUMN]

    print(f"Found {len(ie_columns)} Ireland-related columns")

    # Load full data with only Ireland columns
    df = pd.read_csv(
        file_path,
        usecols=ie_columns,
        parse_dates=[DATE_COLUMN],
        index_col=DATE_COLUMN
    )

    print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")

    return df


def prepare_demand_data(df):
    """
    Prepare the demand data for modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw Ireland data from OPSD

    Returns:
    --------
    pd.DataFrame : Cleaned demand data with target column
    """
    # Check available load columns
    load_columns = [col for col in df.columns if 'load' in col.lower()]
    print(f"Available load columns: {load_columns}")

    # Select the best available load column
    if TARGET_COLUMN in df.columns:
        target_col = TARGET_COLUMN
    elif 'IE_load_actual_entsoe_power_statistics' in df.columns:
        target_col = 'IE_load_actual_entsoe_power_statistics'
    elif load_columns:
        target_col = load_columns[0]
    else:
        raise ValueError("No load column found in the data")

    print(f"Using target column: {target_col}")

    # Create clean dataframe with demand
    demand_df = pd.DataFrame()
    demand_df['demand'] = df[target_col].copy()

    # Add renewable generation features (proxy for weather conditions)
    print("\nAdding weather/generation features...")

    # Wind generation (proxy for wind speed)
    wind_cols = [col for col in df.columns if 'wind' in col.lower() and 'actual' in col.lower()]
    if wind_cols:
        # Combine all wind generation
        demand_df['wind_generation'] = df[wind_cols].sum(axis=1)
        print(f"  Added wind generation from {len(wind_cols)} columns")

    # Solar generation (proxy for solar irradiance)
    solar_cols = [col for col in df.columns if 'solar' in col.lower() and 'actual' in col.lower()]
    if solar_cols:
        demand_df['solar_generation'] = df[solar_cols].sum(axis=1)
        print(f"  Added solar generation from {len(solar_cols)} columns")

    # Total renewable generation
    if 'wind_generation' in demand_df.columns or 'solar_generation' in demand_df.columns:
        demand_df['renewable_generation'] = demand_df.get('wind_generation', 0) + demand_df.get('solar_generation', 0)

    # Renewable share (if we have generation data)
    if 'renewable_generation' in demand_df.columns:
        demand_df['renewable_share'] = demand_df['renewable_generation'] / demand_df['demand'].clip(lower=1)

    # Load forecast (can be useful feature)
    forecast_cols = [col for col in df.columns if 'forecast' in col.lower() and 'load' in col.lower()]
    if forecast_cols:
        demand_df['load_forecast'] = df[forecast_cols[0]]
        print(f"  Added load forecast: {forecast_cols[0]}")

    # Remove rows with missing demand
    initial_len = len(demand_df)
    demand_df = demand_df.dropna(subset=['demand'])
    print(f"Removed {initial_len - len(demand_df)} rows with missing demand")

    # Basic statistics
    print("\nDemand Statistics:")
    print(demand_df['demand'].describe())

    return demand_df


def load_and_prepare_data(force_download=False):
    """
    Complete data loading pipeline.

    Parameters:
    -----------
    force_download : bool
        If True, force re-download of data

    Returns:
    --------
    pd.DataFrame : Prepared demand data ready for feature engineering
    """
    # Download if needed
    file_path = download_opsd_data(force_download)

    # Load Ireland data
    df = load_ireland_data(file_path)

    # Prepare demand data
    demand_df = prepare_demand_data(df)

    # Save processed data
    processed_path = DATA_DIR / "ireland_demand_processed.csv"
    demand_df.to_csv(processed_path)
    print(f"\nProcessed data saved to {processed_path}")

    return demand_df


if __name__ == "__main__":
    # Test data loading
    df = load_and_prepare_data()
    print("\nData loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
