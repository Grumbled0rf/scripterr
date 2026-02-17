"""
Machine Learning Models for Electricity Demand Forecasting
Includes: ARIMA, Random Forest, XGBoost, LSTM, and Hybrid models
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
import joblib

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    XGBOOST_PARAMS, RF_PARAMS, LSTM_SEQUENCE_LENGTH,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_PATIENCE, RANDOM_STATE, MODELS_DIR
)


# =============================================================================
# BASELINE MODELS
# =============================================================================

class ARIMAModel:
    """SARIMA model for baseline comparison."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def fit(self, train_data):
        """Fit SARIMA model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        print("Fitting SARIMA model...")
        print(f"Order: {self.order}, Seasonal Order: {self.seasonal_order}")

        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        print("SARIMA model fitted successfully")

        return self

    def predict(self, steps):
        """Generate forecasts."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def save(self, path):
        """Save model."""
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load model."""
        return joblib.load(path)


# =============================================================================
# TREE-BASED MODELS
# =============================================================================

class RandomForestModel:
    """Random Forest model for demand forecasting."""

    def __init__(self, **params):
        from sklearn.ensemble import RandomForestRegressor

        self.params = {**RF_PARAMS, **params}
        self.model = RandomForestRegressor(**self.params)
        self.feature_importance = None

    def fit(self, X_train, y_train):
        """Fit Random Forest model."""
        print("Fitting Random Forest model...")
        self.model.fit(X_train, y_train)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Random Forest model fitted successfully")
        return self

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, top_n=20):
        """Get top N important features."""
        return self.feature_importance.head(top_n)

    def save(self, path):
        """Save model."""
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load model."""
        return joblib.load(path)


class XGBoostModel:
    """XGBoost model for demand forecasting."""

    def __init__(self, **params):
        from xgboost import XGBRegressor

        self.params = {**XGBOOST_PARAMS, **params}
        self.model = XGBRegressor(**self.params)
        self.feature_importance = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit XGBoost model with optional early stopping."""
        print("Fitting XGBoost model...")

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("XGBoost model fitted successfully")
        return self

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, top_n=20):
        """Get top N important features."""
        return self.feature_importance.head(top_n)

    def save(self, path):
        """Save model."""
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load model."""
        return joblib.load(path)


# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

class LSTMModel:
    """LSTM model for sequential demand forecasting."""

    def __init__(self, input_shape, units=LSTM_UNITS, dropout=LSTM_DROPOUT):
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.model = None
        self.history = None

    def build_model(self):
        """Build LSTM architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")

        model = Sequential([
            LSTM(self.units[0], return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(self.dropout),

            LSTM(self.units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout),

            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        print("LSTM model built successfully")
        print(model.summary())

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, patience=LSTM_PATIENCE):
        """Train LSTM model."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        print("Training LSTM model...")

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        print("LSTM model trained successfully")
        return self

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path):
        """Save model."""
        self.model.save(path)

    @staticmethod
    def load(path):
        """Load model."""
        from tensorflow.keras.models import load_model
        model = LSTMModel(input_shape=None)
        model.model = load_model(path)
        return model


# =============================================================================
# HYBRID MODEL
# =============================================================================

class HybridModel:
    """
    Hybrid model combining XGBoost and LSTM.

    Architecture:
    1. XGBoost captures non-linear feature interactions
    2. LSTM captures sequential temporal patterns
    3. Meta-learner combines both outputs
    """

    def __init__(self, sequence_length=LSTM_SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.xgboost_model = None
        self.lstm_model = None
        self.meta_model = None
        self.feature_columns = None
        self.target_scaler = None

    def prepare_lstm_sequences(self, data, target):
        """Prepare sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_columns=None):
        """
        Fit the hybrid model.

        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation target
        feature_columns : list, optional
            List of feature column names
        """
        print("=" * 50)
        print("TRAINING HYBRID MODEL")
        print("=" * 50)

        self.feature_columns = feature_columns or list(range(X_train.shape[1]))

        # Step 1: Train XGBoost
        print("\n[1/3] Training XGBoost component...")
        self.xgboost_model = XGBoostModel()
        self.xgboost_model.fit(X_train, y_train, X_val, y_val)

        # Get XGBoost predictions for training meta-learner
        xgb_train_pred = self.xgboost_model.predict(X_train)
        xgb_residuals = y_train - xgb_train_pred

        # Step 2: Train LSTM on residuals (to capture patterns XGBoost missed)
        print("\n[2/3] Training LSTM component on residuals...")

        # Prepare sequences from features for LSTM
        X_array = X_train.values if hasattr(X_train, 'values') else X_train

        # Use a subset of important features for LSTM to reduce complexity
        n_features_lstm = min(20, X_array.shape[1])
        top_features_idx = self.xgboost_model.feature_importance.head(n_features_lstm).index.tolist()

        if hasattr(X_train, 'columns'):
            top_feature_names = self.xgboost_model.feature_importance.head(n_features_lstm)['feature'].tolist()
            top_features_idx = [list(X_train.columns).index(f) for f in top_feature_names if f in X_train.columns]

        X_lstm_features = X_array[:, top_features_idx] if len(top_features_idx) > 0 else X_array[:, :n_features_lstm]

        # Prepare sequences
        X_lstm, y_lstm = self.prepare_lstm_sequences(X_lstm_features, xgb_residuals)

        if len(X_lstm) > 0:
            input_shape = (X_lstm.shape[1], X_lstm.shape[2])
            self.lstm_model = LSTMModel(input_shape=input_shape)
            self.lstm_model.build_model()

            # Prepare validation data for LSTM
            X_val_lstm, y_val_lstm = None, None
            if X_val is not None and y_val is not None:
                X_val_array = X_val.values if hasattr(X_val, 'values') else X_val
                X_val_lstm_features = X_val_array[:, top_features_idx] if len(top_features_idx) > 0 else X_val_array[:, :n_features_lstm]
                xgb_val_pred = self.xgboost_model.predict(X_val)
                xgb_val_residuals = y_val - xgb_val_pred
                X_val_lstm, y_val_lstm = self.prepare_lstm_sequences(X_val_lstm_features, xgb_val_residuals)

            self.lstm_model.fit(X_lstm, y_lstm, X_val_lstm, y_val_lstm, epochs=50)
        else:
            print("Warning: Not enough data for LSTM sequences. Using XGBoost only.")
            self.lstm_model = None

        # Step 3: Train meta-learner to combine predictions
        print("\n[3/3] Training meta-learner...")
        from sklearn.linear_model import Ridge

        # Get aligned predictions
        if self.lstm_model is not None:
            lstm_pred = np.zeros(len(y_train))
            lstm_start_idx = self.sequence_length
            lstm_pred[lstm_start_idx:] = self.lstm_model.predict(X_lstm)

            meta_features = np.column_stack([xgb_train_pred, lstm_pred])
        else:
            meta_features = xgb_train_pred.reshape(-1, 1)

        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y_train)

        print("\nHybrid model training complete!")

        # Store feature indices for LSTM
        self.lstm_feature_indices = top_features_idx if len(top_features_idx) > 0 else list(range(n_features_lstm))

        return self

    def predict(self, X):
        """Generate predictions using hybrid model."""
        # XGBoost predictions
        xgb_pred = self.xgboost_model.predict(X)

        if self.lstm_model is not None:
            # Prepare LSTM input
            X_array = X.values if hasattr(X, 'values') else X
            X_lstm_features = X_array[:, self.lstm_feature_indices]

            # LSTM predictions (need sequence context)
            if len(X_lstm_features) >= self.sequence_length:
                X_lstm, _ = self.prepare_lstm_sequences(X_lstm_features, np.zeros(len(X_lstm_features)))
                lstm_pred = np.zeros(len(X))

                if len(X_lstm) > 0:
                    lstm_pred[self.sequence_length:] = self.lstm_model.predict(X_lstm)

                meta_features = np.column_stack([xgb_pred, lstm_pred])
            else:
                meta_features = xgb_pred.reshape(-1, 1)
        else:
            meta_features = xgb_pred.reshape(-1, 1)

        # Meta-learner final prediction
        final_pred = self.meta_model.predict(meta_features)

        return final_pred

    def get_feature_importance(self, top_n=20):
        """Get XGBoost feature importance."""
        return self.xgboost_model.get_feature_importance(top_n)

    def save(self, path):
        """Save hybrid model components."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.xgboost_model.save(path / 'xgboost_model.pkl')
        if self.lstm_model is not None:
            self.lstm_model.save(path / 'lstm_model.h5')
        joblib.dump(self.meta_model, path / 'meta_model.pkl')
        joblib.dump({
            'feature_columns': self.feature_columns,
            'lstm_feature_indices': self.lstm_feature_indices,
            'sequence_length': self.sequence_length
        }, path / 'config.pkl')

        print(f"Hybrid model saved to {path}")

    @staticmethod
    def load(path):
        """Load hybrid model."""
        path = Path(path)

        model = HybridModel()
        model.xgboost_model = XGBoostModel.load(path / 'xgboost_model.pkl')

        lstm_path = path / 'lstm_model.h5'
        if lstm_path.exists():
            model.lstm_model = LSTMModel.load(lstm_path)
        else:
            model.lstm_model = None

        model.meta_model = joblib.load(path / 'meta_model.pkl')

        config = joblib.load(path / 'config.pkl')
        model.feature_columns = config['feature_columns']
        model.lstm_feature_indices = config['lstm_feature_indices']
        model.sequence_length = config['sequence_length']

        return model


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_columns):
    """
    Train all models and return them in a dictionary.

    Returns:
    --------
    dict : Dictionary of trained models
    """
    models = {}

    # Random Forest
    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST")
    print("=" * 50)
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model

    # XGBoost
    print("\n" + "=" * 50)
    print("TRAINING XGBOOST")
    print("=" * 50)
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    models['xgboost'] = xgb_model

    # Hybrid Model
    print("\n" + "=" * 50)
    print("TRAINING HYBRID MODEL")
    print("=" * 50)
    hybrid_model = HybridModel()
    hybrid_model.fit(X_train, y_train, X_val, y_val, feature_columns)
    models['hybrid'] = hybrid_model

    return models
