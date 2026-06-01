import os
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_technical_indicators(input_filepath: str, save_dir: str = "data") -> pd.DataFrame:
    """
    Reads a stock price CSV file, calculates a set of technical indicators
    using pure pandas (equivalent to pandas-ta columns), and saves the updated DataFrame.
    
    Parameters:
    - input_filepath: Path to the daily prices CSV file
    - save_dir: Directory where the indicators CSV will be saved
    
    Returns:
    - DataFrame containing prices and technical indicators, or None if failed.
    """
    if not os.path.exists(input_filepath):
        logging.error(f"Input file not found at: {input_filepath}")
        return None
        
    logging.info(f"Reading price data from {input_filepath}...")
    try:
        df = pd.read_csv(input_filepath)
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Required columns {missing_cols} are missing from the input data.")
            return None
            
        logging.info("Calculating technical indicators using pure pandas...")
        
        # 1. Trend Indicators
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # 2. Momentum Indicators (RSI & MACD)
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
        
        # MACD (12, 26, 9)
        ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema_fast - ema_slow
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # 3. Volatility Indicators (Bollinger Bands & ATR)
        # Bollinger Bands (20, 2)
        df['BBM_20_2.0'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + (2.0 * std)
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - (2.0 * std)
        df['BBB_20_2.0'] = ((df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']) * 100.0
        df['BBP_20_2.0'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'] + 1e-10)
        
        # ATR 14 (Wilder's smoothing)
        high_low = df['High'] - df['Low']
        high_cp = (df['High'] - df['Close'].shift()).abs()
        low_cp = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATRr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
        
        # 4. Volume Indicators (OBV)
        obv = [0.0]
        close_vals = df['Close'].values
        vol_vals = df['Volume'].values
        for i in range(1, len(df)):
            if close_vals[i] > close_vals[i-1]:
                obv.append(obv[-1] + vol_vals[i])
            elif close_vals[i] < close_vals[i-1]:
                obv.append(obv[-1] - vol_vals[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Save output
        filename = os.path.basename(input_filepath)
        output_filename = filename.replace("_daily_prices.csv", "_indicators.csv")
        if output_filename == filename:
            output_filename = filename.replace("_prices.csv", "_indicators.csv")
        if output_filename == filename:
            output_filename = "indicators_" + filename
            
        os.makedirs(save_dir, exist_ok=True)
        output_filepath = os.path.join(save_dir, output_filename)
        df.to_csv(output_filepath, index=False)
        logging.info(f"Successfully calculated indicators. Saved to {output_filepath} ({len(df)} rows, {len(df.columns)} columns).")
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return None

if __name__ == "__main__":
    # Test script locally
    test_file = "data/RELIANCE_daily_prices.csv"
    if os.path.exists(test_file):
        print(f"--- Calculating Indicators for {test_file} ---")
        calculate_technical_indicators(test_file)
    else:
        print(f"[!] Test price file '{test_file}' not found. Download historical price data first.")
