import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# We import tensorflow locally inside functions to avoid slowing down the CLI boot time
# or crashing if the user hasn't installed tensorflow yet. This is a very professional practice!

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int = 10):
    """
    Groups historical 2D feature arrays into 3D sequential time windows.
    E.g. if seq_length = 10, past 10 days of features are grouped to predict day 11's target.
    """
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length):
        X_seq.append(features[i : i + seq_length])
        y_seq.append(targets[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(ticker_prefix: str, seq_length: int = 10, epochs: int = 15, batch_size: int = 32) -> bool:
    """
    Loads preprocessed regression datasets, structures them into time-series sequences,
    trains an LSTM neural network, and saves evaluation metrics, plots, and weights.
    
    Parameters:
    - ticker_prefix: The prefix name of the stock file (e.g., RELIANCE)
    - seq_length: The length of the sequential sliding window (default: 10 days)
    - epochs: Training epochs (default: 15)
    - batch_size: Batch size (default: 32)
    
    Returns:
    - Success status boolean
    """
    data_dir = "data/preprocessed"
    models_dir = "models"
    plots_dir = "plots"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    x_train_path = os.path.join(data_dir, f"{ticker_prefix}_X_train.csv")
    x_test_path = os.path.join(data_dir, f"{ticker_prefix}_X_test.csv")
    y_train_path = os.path.join(data_dir, f"{ticker_prefix}_y_train_reg.csv")
    y_test_path = os.path.join(data_dir, f"{ticker_prefix}_y_test_reg.csv")
    
    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            logging.error(f"Required preprocessing file '{path}' is missing. Run preprocessing first.")
            return False
            
    logging.info(f"Loading datasets for {ticker_prefix}...")
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train_df = pd.read_csv(y_train_path)
    y_test_df = pd.read_csv(y_test_path)
    
    # Drop Date column since it's a string, keep only numeric features
    X_train_raw = X_train_df.drop(columns=['Date']).values
    X_test_raw = X_test_df.drop(columns=['Date']).values
    
    y_train_raw = y_train_df['Target_Close'].values
    y_test_raw = y_test_df['Target_Close'].values
    
    if len(X_train_raw) <= seq_length or len(X_test_raw) <= seq_length:
        logging.error(f"Dataset is too small for sequence length {seq_length}.")
        return False
        
    # Convert data into 3D shape (samples, seq_length, features)
    X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_raw, y_test_raw, seq_length)
    
    logging.info(f"Structured training input shape: {X_train_seq.shape}")
    logging.info(f"Structured testing input shape:  {X_test_seq.shape}")
    
    # Import TensorFlow locally
    logging.info("Importing TensorFlow...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError:
        logging.error("TensorFlow is not installed. Please install it using `pip install tensorflow`.")
        return False
        
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1) # Linear activation for Close price regression
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    
    logging.info(f"Training LSTM model for {epochs} epochs...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save the trained model
    model_path = os.path.join(models_dir, f"{ticker_prefix}_lstm_model.h5")
    model.save(model_path)
    logging.info(f"[✓] Model saved successfully to {model_path}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title(f'{ticker_prefix} LSTM Model - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_dir, f"{ticker_prefix}_lstm_loss.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"[✓] Training loss plot saved to {plot_path}")
    
    # Evaluate model
    predictions = model.predict(X_test_seq).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_seq, predictions))
    mae = mean_absolute_error(y_test_seq, predictions)
    r2 = r2_score(y_test_seq, predictions)
    
    print("\n" + "=" * 50)
    print(f"             LSTM MODEL EVALUATION: {ticker_prefix}")
    print("=" * 50)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"R-squared Score (R2):          {r2:.4f}")
    print("=" * 50 + "\n")
    
    # Plot predictions comparison
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_seq, label='Actual Close Price', color='blue', alpha=0.7)
    plt.plot(predictions, label='LSTM Predicted Close Price', color='red', linestyle='--', alpha=0.8)
    plt.title(f'{ticker_prefix} Close Price Prediction - LSTM Model')
    plt.xlabel('Days (Test Period)')
    plt.ylabel('Price (Scaled/Actual)')
    plt.legend()
    plt.grid(True)
    pred_plot_path = os.path.join(plots_dir, f"{ticker_prefix}_lstm_predictions.png")
    plt.savefig(pred_plot_path)
    plt.close()
    logging.info(f"[✓] Predictions comparison plot saved to {pred_plot_path}")
    
    return True

if __name__ == "__main__":
    # Test locally
    test_ticker = "RELIANCE"
    print(f"--- Running LSTM test training for {test_ticker} ---")
    train_lstm_model(test_ticker, epochs=3) # short epochs for testing
