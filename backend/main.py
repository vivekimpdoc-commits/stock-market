import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_prices import fetch_historical_daily, fetch_intraday
from fetch_fundamentals import fetch_financial_sheets, fetch_key_valuation_metrics
from fetch_sentiment import fetch_and_analyze_sentiment
from indicators import calculate_technical_indicators
from data_prep import prepare_ml_data
from model_lstm import train_lstm_model
from model_xgboost import train_xgboost_model
from model_finbert import run_finbert_sentiment
from portfolio_opt import optimize_portfolio
from backtest import run_backtest
from run_pipeline import run_end_to_end_pipeline

def clean_ticker_input(ticker: str) -> str:
    """
    Cleans and normalizes the ticker input.
    If the ticker does not contain an extension (e.g., RELIANCE),
    we append '.NS' (NSE) as a default helper for Indian markets.
    """
    ticker = ticker.strip().upper()
    if ticker and not ticker.startswith("^") and "." not in ticker:
        # Default to NSE if no suffix provided
        ticker = f"{ticker}.NS"
        print(f"[*] Suffix missing. Auto-converted to: {ticker}")
    return ticker

def get_clean_name(ticker: str) -> str:
    """
    Returns the clean ticker name for checking file existence (e.g. RELIANCE).
    """
    return ticker.replace("^", "").replace(".NS", "").replace(".BO", "")

def run_interactive_menu():
    print("=" * 75)
    print("   INDIAN STOCK MARKET DATA ACQUISITION, AI CORE, BACKTESTING & API APP   ")
    print("=" * 75)
    print("Choose an option:")
    print("1. Fetch Historical & Intraday Price Data")
    print("2. Fetch Company Fundamentals (Balance Sheet, P&L, Key Metrics)")
    print("3. Fetch Live News & Analyze Sentiment (VADER)")
    print("4. Calculate Technical Indicators (RSI, MACD, SMAs, OBV, etc.)")
    print("5. Preprocess & Scale Data for Machine Learning (Train/Test Split)")
    print("6. Train LSTM Price Prediction Model (Deep Learning Time-Series)")
    print("7. Train XGBoost/RF Price Direction Classifier (Trend Forecasting)")
    print("8. Run FinBERT Sentiment Analysis (HF Financial Transformer)")
    print("9. Optimize Stock Portfolio Allocation (Modern Portfolio Theory)")
    print("10. Run AI Strategy Backtesting (Simulation of Profits/Losses)")
    print("11. Run End-to-End AI Pipeline (Download, Train, and Backtest a stock)")
    print("12. Start FastAPI API Server (Launch Web API hosting for Mobile App)")
    print("13. Exit")
    print("-" * 75)
    
    choice = input("Enter choice (1-13): ").strip()
    
    if choice == "1":
        ticker = input("Enter Stock Ticker (e.g., RELIANCE, TCS, or ^NSEI for Nifty): ").strip()
        ticker = clean_ticker_input(ticker)
        if not ticker:
            print("[!] Invalid ticker.")
            return
            
        print("\n--- Fetching Prices ---")
        start_date = input("Enter start date (YYYY-MM-DD) [default: 2011-01-01]: ").strip()
        if not start_date:
            start_date = "2011-01-01"
            
        fetch_historical_daily(ticker, start_date=start_date)
        
        fetch_intra = input("Do you also want to fetch 5-minute intraday data for the last 5 days? (y/n): ").strip().lower()
        if fetch_intra == 'y':
            fetch_intraday(ticker, interval="5m", period="5d")
            
    elif choice == "2":
        ticker = input("Enter Stock Ticker (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        if not ticker:
            print("[!] Invalid ticker.")
            return
            
        print("\n--- Fetching Fundamentals ---")
        fetch_financial_sheets(ticker)
        fetch_key_valuation_metrics(ticker)
        
    elif choice == "3":
        keyword = input("Enter keyword to filter news (e.g., 'Reliance', 'Market' or leave blank for all): ").strip()
        print("\n--- Fetching News & Sentiment Analysis ---")
        if keyword:
            fetch_and_analyze_sentiment(keyword_filter=keyword)
        else:
            fetch_and_analyze_sentiment()
            
    elif choice == "4":
        ticker = input("Enter Stock Ticker to calculate indicators for (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        clean_name = get_clean_name(ticker)
        price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
        
        if not os.path.exists(price_file):
            print(f"[!] Historical daily prices file '{price_file}' not found.")
            print("[*] Downloading prices first...")
            fetch_historical_daily(ticker, start_date="2011-01-01")
            
        print("\n--- Calculating Technical Indicators ---")
        calculate_technical_indicators(price_file)
        
    elif choice == "5":
        ticker = input("Enter Stock Ticker to prepare ML datasets for (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        clean_name = get_clean_name(ticker)
        indicators_file = os.path.join("data", f"{clean_name}_indicators.csv")
        
        if not os.path.exists(indicators_file):
            print(f"[!] Indicators file '{indicators_file}' not found.")
            price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
            if not os.path.exists(price_file):
                print(f"[*] Price file '{price_file}' not found either. Fetching prices...")
                fetch_historical_daily(ticker, start_date="2011-01-01")
            print("[*] Calculating indicators...")
            calculate_technical_indicators(price_file)
            
        print("\n--- Preparing Machine Learning Datasets ---")
        prepare_ml_data(indicators_file)
        
    elif choice == "6":
        ticker = input("Enter Stock Ticker to train LSTM on (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        clean_name = get_clean_name(ticker)
        
        x_train = os.path.join("data/preprocessed", f"{clean_name}_X_train.csv")
        if not os.path.exists(x_train):
            print(f"[*] Preprocessed ML data for {clean_name} not found. Automatically running data prep...")
            # Ensure indicators are calculated
            ind_file = os.path.join("data", f"{clean_name}_indicators.csv")
            if not os.path.exists(ind_file):
                pr_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
                if not os.path.exists(pr_file):
                    fetch_historical_daily(ticker, start_date="2011-01-01")
                calculate_technical_indicators(pr_file)
            prepare_ml_data(ind_file)
            
        print("\n--- Training LSTM Time-Series Model ---")
        epochs_input = input("Enter number of training epochs [default: 15]: ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else 15
        train_lstm_model(clean_name, epochs=epochs)
        
    elif choice == "7":
        ticker = input("Enter Stock Ticker to train XGBoost on (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        clean_name = get_clean_name(ticker)
        
        x_train = os.path.join("data/preprocessed", f"{clean_name}_X_train.csv")
        if not os.path.exists(x_train):
            print(f"[*] Preprocessed ML data for {clean_name} not found. Automatically running data prep...")
            ind_file = os.path.join("data", f"{clean_name}_indicators.csv")
            if not os.path.exists(ind_file):
                pr_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
                if not os.path.exists(pr_file):
                    fetch_historical_daily(ticker, start_date="2011-01-01")
                calculate_technical_indicators(pr_file)
            prepare_ml_data(ind_file)
            
        print("\n--- Training XGBoost Classifier Model ---")
        train_xgboost_model(clean_name)
        
    elif choice == "8":
        print("\n--- Running FinBERT Sentiment Analysis ---")
        keyword = input("Enter keyword filter used for news (e.g., Reliance) or leave blank for general news: ").strip()
        
        if keyword:
            news_file = os.path.join("data", f"market_sentiment_{keyword}.csv")
        else:
            news_file = os.path.join("data", "market_sentiment.csv")
            
        if not os.path.exists(news_file):
            print(f"[!] News file '{news_file}' not found.")
            print("[*] Fetching general headlines first...")
            fetch_and_analyze_sentiment(keyword_filter=keyword if keyword else None)
            
        run_finbert_sentiment(news_file)
        
    elif choice == "9":
        print("\n--- Modern Portfolio Theory (MPT) Optimization ---")
        stocks_str = input("Enter Stock Tickers separated by comma (e.g., RELIANCE, TCS, INFY, HDFCBANK): ").strip()
        
        if not stocks_str:
            stocks_list = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
            print(f"[*] No tickers input. Optimizing default portfolio: {stocks_list}")
        else:
            stocks_list = [clean_ticker_input(s.strip()) for s in stocks_str.split(",") if s.strip()]
            
        optimize_portfolio(stocks_list)
        
    elif choice == "10":
        ticker = input("Enter Stock Ticker to backtest on (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        clean_name = get_clean_name(ticker)
        
        xgb_path = os.path.join("models", f"{clean_name}_xgboost_model.json")
        rf_path = os.path.join("models", f"{clean_name}_randomforest_model.pkl")
        
        if not os.path.exists(xgb_path) and not os.path.exists(rf_path):
            print(f"[*] Model not found for {clean_name}. Automatically training classifier model first...")
            train_xgboost_model(clean_name)
            
        print("\n--- Running AI Strategy Backtesting ---")
        cash_input = input("Enter starting capital (INR) [default: 100000.0]: ").strip()
        cash = float(cash_input) if cash_input.replace(".", "", 1).isdigit() else 100000.0
        
        run_backtest(clean_name, initial_cash=cash)
        
    elif choice == "11":
        ticker = input("Enter Stock Ticker to run end-to-end pipeline (e.g., RELIANCE, TCS): ").strip()
        start_date_input = input("Enter start date (YYYY-MM-DD) [default: 2011-01-01]: ").strip()
        start_date = start_date_input if start_date_input else "2011-01-01"
        
        cash_input = input("Enter starting capital (INR) [default: 100000.0]: ").strip()
        cash = float(cash_input) if cash_input.replace(".", "", 1).isdigit() else 100000.0
        
        run_end_to_end_pipeline(ticker, start_date=start_date, initial_capital=cash)
        
    elif choice == "12":
        print("\n--- Starting FastAPI API Web Server ---")
        print("[*] Interactive API docs will be available at: http://127.0.0.1:8000/docs")
        print("[*] Press Ctrl+C in terminal to stop the server.")
        try:
            import uvicorn
            uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
        except ImportError:
            print("[!] Uvicorn is not installed. Please install it using: pip install uvicorn fastapi")
        except Exception as e:
            print(f"[!] Error launching API server: {e}")
            
    elif choice == "13":
        print("Exiting pipeline. Happy trading!")
        sys.exit(0)
    else:
        print("[!] Invalid option. Please select between 1 and 13.")

if __name__ == "__main__":
    try:
        while True:
            run_interactive_menu()
            print("\n" + "=" * 75 + "\n")
    except KeyboardInterrupt:
        print("\nExiting pipeline. Goodbye!")
