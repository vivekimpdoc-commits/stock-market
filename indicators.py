import os
import pandas as pd
import pandas_ta as ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_technical_indicators(input_filepath: str, save_dir: str = "data") -> pd.DataFrame:
    """
    Reads a stock price CSV file, calculates a set of technical indicators
    using pandas-ta, and saves the updated DataFrame to a new CSV file.
    
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
            
        # Ensure 'Date' is set as the datetime index for pandas-ta calculations
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            logging.warning("No 'Date' column found. Proceeding with default indexing.")
            
        logging.info("Calculating technical indicators...")
        
        # 1. Trend Indicators
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ema(length=20, append=True)
        
        # 2. Momentum Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # 3. Volatility Indicators
        df.ta.bbands(length=20, std=2.0, append=True)
        df.ta.atr(length=14, append=True)
        
        # 4. Volume Indicators
        df.ta.obv(append=True)
        
        # Reset index back so Date becomes a normal column again
        df.reset_index(inplace=True)
        
        # Format the Date back to standard string for CSV
        if 'Date' in df.columns:
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
        # Save output
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(input_filepath)
        output_filename = filename.replace("_prices.csv", "_indicators.csv")
        if output_filename == filename:  # Fallback if name format is unexpected
            output_filename = "indicators_" + filename
            
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
