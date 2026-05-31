import os
import sys
from fetch_prices import fetch_historical_daily, fetch_intraday
from fetch_fundamentals import fetch_financial_sheets, fetch_key_valuation_metrics
from fetch_sentiment import fetch_and_analyze_sentiment

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

def run_interactive_menu():
    print("=" * 60)
    print("      INDIAN STOCK MARKET DATA ACQUISITION PIPELINE      ")
    print("=" * 60)
    print("Choose an option:")
    print("1. Fetch Historical & Intraday Price Data")
    print("2. Fetch Company Fundamentals (Balance Sheet, P&L, Key Metrics)")
    print("3. Fetch Live News & Analyze Sentiment")
    print("4. Run Complete Pipeline (All of the above for a company)")
    print("5. Exit")
    print("-" * 60)
    
    choice = input("Enter choice (1-5): ").strip()
    
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
        ticker = input("Enter Stock Ticker (e.g., RELIANCE, TCS): ").strip()
        ticker = clean_ticker_input(ticker)
        if not ticker:
            print("[!] Invalid ticker.")
            return
            
        print(f"\n=== RUNNING COMPLETE PIPELINE FOR {ticker} ===")
        
        # 1. Prices
        print("\n[Step 1/3] Fetching 15-year daily historical prices...")
        fetch_historical_daily(ticker, start_date="2011-01-01")
        print("Fetching 5-minute intraday prices for the last 5 days...")
        fetch_intraday(ticker, interval="5m", period="5d")
        
        # 2. Fundamentals
        print("\n[Step 2/3] Fetching financial statements & key metrics...")
        fetch_financial_sheets(ticker)
        fetch_key_valuation_metrics(ticker)
        
        # 3. Sentiments
        # We will filter news matching the clean ticker name (e.g., RELIANCE for RELIANCE.NS)
        clean_name = ticker.replace(".NS", "").replace(".BO", "")
        print(f"\n[Step 3/3] Fetching news and filtering for keyword '{clean_name}'...")
        fetch_and_analyze_sentiment(keyword_filter=clean_name)
        
        print("\n[✓] Pipeline execution finished! Check the 'data/' folder.")
        
    elif choice == "5":
        print("Exiting pipeline. Happy trading!")
        sys.exit(0)
    else:
        print("[!] Invalid option. Please select between 1 and 5.")

if __name__ == "__main__":
    try:
        while True:
            run_interactive_menu()
            print("\n" + "=" * 60 + "\n")
    except KeyboardInterrupt:
        print("\nExiting pipeline. Goodbye!")
