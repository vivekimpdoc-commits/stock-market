import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_ticker_name(ticker: str) -> str:
    """
    Cleans ticker name for filename usage (e.g., RELIANCE.NS -> RELIANCE).
    """
    return ticker.replace("^", "").replace(".NS", "").replace(".BO", "")

def fetch_historical_from_nse_website(symbol: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Fetches historical stock prices directly from the official NSE India website (nseindia.com).
    Queries in chunks of 50 days (NSE API limit) and merges the results.
    """
    symbol = symbol.strip().upper()
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError as e:
        logging.error(f"[-] Date formatting error in direct NSE fetcher: {e}")
        return None
        
    logging.info(f"[*] Attempting direct fetch from official NSE India website for symbol: {symbol}...")
    
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
    }
    
    # Visit homepage first to set cookies
    try:
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
    except Exception as e:
        logging.warning(f"[-] Failed to establish session with nseindia.com: {e}")
        return None
        
    all_data = []
    current_start = start_dt
    
    # Direct scraping is slow due to rate-limiting and session timeouts.
    # Restrict to last 2 years for direct scraping to prevent IP block.
    # If the user requests a larger historical range, we fallback to Yahoo Finance (which is an official NSE mirror).
    total_days = (end_dt - start_dt).days
    if total_days > 730:
        logging.info(f"[*] Requested range of {total_days} days is large. To protect from rate limiting, using official NSE mirror (Yahoo Finance).")
        return None
        
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=49), end_dt)
        from_str = current_start.strftime("%d-%m-%Y")
        to_str = current_end.strftime("%d-%m-%Y")
        
        # Historical equity details API endpoint
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={from_str}&to={to_str}"
        logging.info(f"[*] Querying NSE website: {from_str} to {to_str}...")
        
        try:
            response = session.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                res_json = response.json()
                if "data" in res_json and len(res_json["data"]) > 0:
                    all_data.extend(res_json["data"])
            elif response.status_code == 403:
                logging.warning("[-] NSE website returned 403 Forbidden. IP may be rate-limited.")
                return None
            else:
                logging.warning(f"[-] NSE API returned code {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[-] Error querying direct NSE API chunk: {e}")
            return None
            
        current_start = current_end + timedelta(days=1)
        time.sleep(0.5) # Sleep to avoid rate limiting
        
    if not all_data:
        return None
        
    # Build dataframe
    records = []
    for item in all_data:
        # In the response of NSE historical equity endpoint, the keys are:
        # CH_TIMESTAMP, CH_OPENING_PRICE, CH_TRADE_HIGH_PRICE, CH_TRADE_LOW_PRICE, CH_CLOSING_PRICE, CH_TOT_TRADED_QTY
        records.append({
            "Date": item.get("CH_TIMESTAMP"),
            "Open": item.get("CH_OPENING_PRICE"),
            "High": item.get("CH_TRADE_HIGH_PRICE"),
            "Low": item.get("CH_TRADE_LOW_PRICE"),
            "Close": item.get("CH_CLOSING_PRICE"),
            "Adj Close": item.get("CH_CLOSING_PRICE"),
            "Volume": item.get("CH_TOT_TRADED_QTY")
        })
        
    df = pd.DataFrame(records)
    
    # Parse Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Cast numeric columns
    num_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Clean rows
    df = df.dropna(subset=['Close'])
    logging.info(f"[OK] Successfully retrieved {len(df)} rows directly from nseindia.com!")
    return df

def fetch_historical_daily(ticker: str, start_date: str = "2011-01-01", end_date: str = None, save_dir: str = "data") -> pd.DataFrame:
    """
    Fetches historical daily stock prices (Open, High, Low, Close, Adj Close, Volume) for a given ticker.
    Maintains a dual search: attempts to fetch directly from nseindia.com, with a robust fallback to 
    Yahoo Finance (.NS suffix representing the National Stock Exchange of India).
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    clean_name = clean_ticker_name(ticker)
    
    # Try fetching directly from the official NSE website
    df = fetch_historical_from_nse_website(clean_name, start_date, end_date)
    
    if df is not None and not df.empty:
        logging.info("[*] Successfully acquired data directly from the official NSE India website.")
    else:
        logging.info("[*] Direct NSE fetch skipped or failed. Using official NSE mirror (Yahoo Finance)...")
        try:
            # Ensure ticker ends with .NS for NSE India
            nse_ticker = ticker
            if not nse_ticker.startswith("^") and "." not in nse_ticker:
                nse_ticker = f"{nse_ticker}.NS"
                
            stock = yf.Ticker(nse_ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                logging.warning(f"No historical daily data found for {nse_ticker} in the specified range.")
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
                
        except Exception as e:
            logging.error(f"[-] Error fetching stock data from NSE mirror: {e}")
            return None
            
    # Ensure target directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Build path and save
    file_path = os.path.join(save_dir, f"{clean_name}_daily_prices.csv")
    df.to_csv(file_path, index=False)
    logging.info(f"[OK] Saved daily prices to {file_path} ({len(df)} rows).")
    return df

def fetch_intraday(ticker: str, interval: str = "5m", period: str = "5d", save_dir: str = "data") -> pd.DataFrame:
    """
    Fetches historical intraday stock prices for NSE stocks.
    """
    # Ensure ticker ends with .NS for NSE India
    nse_ticker = ticker
    if not nse_ticker.startswith("^") and "." not in nse_ticker:
        nse_ticker = f"{nse_ticker}.NS"
        
    logging.info(f"Fetching intraday data for {nse_ticker} with interval={interval} for period={period}...")
    
    try:
        stock = yf.Ticker(nse_ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            logging.warning(f"No intraday data found for {nse_ticker} with interval={interval} and period={period}.")
            return None
            
        # Format datetime column safely
        df = df.reset_index()
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
        logging.info(f"[OK] Saved intraday prices to {file_path} ({len(df)} rows).")
        return df
        
    except Exception as e:
        logging.error(f"[-] Error fetching intraday data for {nse_ticker}: {e}")
        return None

if __name__ == "__main__":
    test_ticker = "RELIANCE.NS"
    print(f"--- Fetching historical data for {test_ticker} ---")
    fetch_historical_daily(test_ticker, start_date="2024-01-01")
    
    print(f"\n--- Fetching intraday 5m data for {test_ticker} ---")
    fetch_intraday(test_ticker, interval="5m", period="5d")
