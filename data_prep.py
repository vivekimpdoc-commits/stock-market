import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_ml_data(input_filepath: str, split_ratio: float = 0.80, save_dir: str = "data/preprocessed") -> dict:
    """
    Cleans stock data with indicators, creates targets for regression & classification,
    splits chronologically to prevent time leakage, scales features, and saves CSVs.
    
    Parameters:
    - input_filepath: Path to the indicators CSV file (e.g. data/RELIANCE_indicators.csv)
    - split_ratio: Chronological train/test split percentage (default: 80% train, 20% test)
    - save_dir: Directory to save the final training/testing files
    
    Returns:
    - Dict with filepaths of saved CSVs
    """
    if not os.path.exists(input_filepath):
        logging.error(f"Input file not found at: {input_filepath}")
        return None
        
    logging.info(f"Preparing ML data from {input_filepath}...")
    
    try:
        df = pd.read_csv(input_filepath)
        
        if len(df) < 250:
            logging.error("Dataset is too small (needs at least 250 rows for indicators like SMA 200).")
            return None
            
        # Keep track of dates for reference
        dates = df['Date'].values if 'Date' in df.columns else np.arange(len(df))
        
        # Drop columns we don't want as features (e.g., date)
        features_df = df.copy()
        if 'Date' in features_df.columns:
            features_df = features_df.drop(columns=['Date'])
            
        # Target variables creation (Predicting next day's price movement)
        # Shift close price by -1 to get the next day's price
        features_df['Next_Close'] = features_df['Close'].shift(-1)
        features_df['Next_Return'] = features_df['Close'].pct_change().shift(-1)
        # Classification Target: 1 if Next day's Close > Today's Close, else 0 (cast to float to keep NaN for last row)
        features_df['Target_Direction'] = (features_df['Next_Close'] > features_df['Close']).astype(float)
        
        # Clean NaN values
        # Lagging technical indicators (like SMA_200) leave NaNs at the start.
        # Target shifts leave a NaN at the very end row.
        # We drop all rows containing any NaNs. This is critical to avoid:
        # 1. Look-ahead bias (backward filling technical indicators)
        # 2. Fabricated targets (forward filling shifted next-day close prices)
        features_df = features_df.dropna()
        
        # Cast target classification back to integer
        features_df['Target_Direction'] = features_df['Target_Direction'].astype(int)
        
        # Get target columns
        y_regression = features_df['Next_Close']
        y_classification = features_df['Target_Direction']
        
        # Remove target-derived fields from features X
        X = features_df.drop(columns=['Next_Close', 'Next_Return', 'Target_Direction'])
        
        # Get matching dates for the filtered indices
        dates_filtered = dates[features_df.index]
        
        # Chronological Split (No random splitting to avoid look-ahead bias)
        split_idx = int(len(X) * split_ratio)
        
        X_train_raw = X.iloc[:split_idx]
        X_test_raw = X.iloc[split_idx:]
        
        y_train_reg = y_regression.iloc[:split_idx]
        y_test_reg = y_regression.iloc[split_idx:]
        
        y_train_clf = y_classification.iloc[:split_idx]
        y_test_clf = y_classification.iloc[split_idx:]
        
        dates_train = dates_filtered[:split_idx]
        dates_test = dates_filtered[split_idx:]
        
        # Feature Scaling
        # CRITICAL: Fit Scaler ONLY on Training data to prevent information leakage!
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # Convert scaled data back to DataFrame to preserve column names
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        # Add Date column to outputs for tracking and alignment
        X_train_scaled_df.insert(0, 'Date', dates_train)
        X_test_scaled_df.insert(0, 'Date', dates_test)
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.basename(input_filepath).replace("_indicators.csv", "")
        
        # Save the fitted scaler object for production API usage
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        scaler_path = os.path.join(models_dir, f"{filename}_scaler.pkl")
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Saved fitted scaler to {scaler_path}")
        
        saved_paths = {
            "X_train": os.path.join(save_dir, f"{filename}_X_train.csv"),
            "X_test": os.path.join(save_dir, f"{filename}_X_test.csv"),
            "y_train_regression": os.path.join(save_dir, f"{filename}_y_train_reg.csv"),
            "y_test_regression": os.path.join(save_dir, f"{filename}_y_test_reg.csv"),
            "y_train_classification": os.path.join(save_dir, f"{filename}_y_train_clf.csv"),
            "y_test_classification": os.path.join(save_dir, f"{filename}_y_test_clf.csv")
        }
        
        # Save files
        X_train_scaled_df.to_csv(saved_paths["X_train"], index=False)
        X_test_scaled_df.to_csv(saved_paths["X_test"], index=False)
        y_train_reg.to_frame(name='Target_Close').to_csv(saved_paths["y_train_regression"], index=False)
        y_test_reg.to_frame(name='Target_Close').to_csv(saved_paths["y_test_regression"], index=False)
        y_train_clf.to_frame(name='Target_Direction').to_csv(saved_paths["y_train_classification"], index=False)
        y_test_clf.to_frame(name='Target_Direction').to_csv(saved_paths["y_test_classification"], index=False)
        
        logging.info(f"Saved preprocessed datasets to {save_dir}/")
        logging.info(f"Training set: {X_train_scaled_df.shape[0]} rows. Test set: {X_test_scaled_df.shape[0]} rows.")
        
        return saved_paths
        
    except Exception as e:
        logging.error(f"Error preparing machine learning data: {e}")
        return None

if __name__ == "__main__":
    # Test script locally
    test_file = "data/RELIANCE_indicators.csv"
    if os.path.exists(test_file):
        print(f"--- Preparing ML Data for {test_file} ---")
        prepare_ml_data(test_file)
    else:
        print(f"[!] Test indicators file '{test_file}' not found. Run technical indicators calculation first.")
