import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_ticker_name(ticker: str) -> str:
    """
    Cleans ticker name for filename usage (e.g., RELIANCE.NS -> RELIANCE).
    """
    return ticker.replace("^", "").replace(".NS", "").replace(".BO", "")

def fetch_historical_daily(ticker: str, start_date: str = "2011-01-01", end_date: str = None, save_dir: str = "data") -> pd.DataFrame:
    """
    Fetches historical daily stock prices (Open, High, Low, Close, Adj Close, Volume) for a given ticker.
    Indian tickers usually end with '.NS' for NSE or '.BO' for BSE (e.g., RELIANCE.NS, TCS.NS, ^NSEI for Nifty 50).
    
    Parameters:
    - ticker: Ticker symbol (e.g. 'RELIANCE.NS')
    - start_date: Start date string in 'YYYY-MM-DD' format (default: '2011-01-01' for ~15 years)
    - end_date: End date string in 'YYYY-MM-DD' format (default: Today)
    - save_dir: Directory where data will be stored as CSV
    
    Returns:
    - DataFrame of historical daily prices or None if failed.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    logging.info(f"Fetching historical daily data for {ticker} from {start_date} to {end_date}...")
    
    try:
        # Download stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            logging.warning(f"No historical daily data found for {ticker} in the specified range.")
            return None
            
        # Reset index to make Date a column and localize/remove timezone safely
        df = df.reset_index()
        if 'Date' in df.columns:
            dates_converted = pd.to_datetime(df['Date'])
            try:
                if dates_converted.dt.tz is not None:
                    dates_converted = dates_converted.dt.tz_localize(None)
            except AttributeError:
                pass
            df['Date'] = dates_converted
            
        # Ensure target directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Build path and save
        clean_name = clean_ticker_name(ticker)
        file_path = os.path.join(save_dir, f"{clean_name}_daily_prices.csv")
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully saved historical daily prices to {file_path} ({len(df)} rows).")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching historical daily data for {ticker}: {e}")
        return None

def fetch_intraday(ticker: str, interval: str = "5m", period: str = "5d", save_dir: str = "data") -> pd.DataFrame:
    """
    Fetches historical intraday stock prices.
    Note on Yahoo Finance intraday limits:
    - 1m interval data is only available for the last 7 days.
    - 5m, 15m, 30m, 60m data is only available for the last 60 days.
    
    Parameters:
    - ticker: Ticker symbol (e.g. 'RELIANCE.NS')
    - interval: Data interval (e.g. '1m', '5m', '15m', '60m')
    - period: Retrieval range (e.g. '1d', '5d', '7d', '60d')
    - save_dir: Directory where data will be stored as CSV
    
    Returns:
    - DataFrame of intraday prices or None if failed.
    """
    logging.info(f"Fetching intraday data for {ticker} with interval={interval} for period={period}...")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            logging.warning(f"No intraday data found for {ticker} with interval={interval} and period={period}.")
            return None
            
        # Format datetime column safely
        df = df.reset_index()
        # yfinance returns 'Datetime' for intraday data
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        if date_col in df.columns:
            dates_converted = pd.to_datetime(df[date_col])
            try:
                if dates_converted.dt.tz is not None:
                    dates_converted = dates_converted.dt.tz_localize(None)
            except AttributeError:
                pass
            df[date_col] = dates_converted
            
        # Ensure target directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Build path and save
        clean_name = clean_ticker_name(ticker)
        file_path = os.path.join(save_dir, f"{clean_name}_intraday_{interval}.csv")
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully saved intraday prices to {file_path} ({len(df)} rows).")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching intraday data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Test script locally
    test_ticker = "RELIANCE.NS"
    print(f"--- Fetching historical data for {test_ticker} ---")
    fetch_historical_daily(test_ticker, start_date="2024-01-01")
    
    print(f"\n--- Fetching intraday 5m data for {test_ticker} ---")
    fetch_intraday(test_ticker, interval="5m", period="5d")
