import os
import sys
import pandas as pd

def verify():
    print("=" * 60)
    print("         VERIFYING DATA ACQUISITION PIPELINE             ")
    print("=" * 60)
    
    # Check if files exist or try to import modules
    print("[1] Checking module imports...")
    try:
        from fetch_prices import fetch_historical_daily, fetch_intraday
        from fetch_fundamentals import fetch_financial_sheets, fetch_key_valuation_metrics
        from fetch_sentiment import fetch_and_analyze_sentiment
        print("[✓] All modules imported successfully!")
    except ImportError as e:
        print(f"[X] Import error: {e}")
        print("[!] Make sure you installed the dependencies using: pip install -r requirements.txt")
        return
        
    test_ticker = "RELIANCE.NS"
    print(f"\n[2] Fetching historical price data for {test_ticker}...")
    try:
        price_df = fetch_historical_daily(test_ticker, start_date="2025-01-01")
        if price_df is not None and not price_df.empty:
            print(f"[✓] Price data successfully saved. Total rows: {len(price_df)}")
        else:
            print("[X] Failed to fetch price data.")
    except Exception as e:
        print(f"[X] Error fetching prices: {e}")
        
    print(f"\n[3] Fetching company fundamentals for {test_ticker}...")
    try:
        sheets = fetch_financial_sheets(test_ticker)
        metrics = fetch_key_valuation_metrics(test_ticker)
        if sheets or metrics is not None:
            print(f"[✓] Fundamental data successfully saved. Sheets downloaded: {list(sheets.keys())}")
        else:
            print("[X] Failed to fetch fundamentals.")
    except Exception as e:
        print(f"[X] Error fetching fundamentals: {e}")
        
    print("\n[4] Fetching latest live news headlines & analyzing sentiments...")
    try:
        sentiment_df = fetch_and_analyze_sentiment(keyword_filter="Reliance")
        if sentiment_df is not None and not sentiment_df.empty:
            print(f"[✓] Sentiment data successfully saved. Headlines analyzed: {len(sentiment_df)}")
            print("\nSample Sentiment Data:")
            print(sentiment_df[['source', 'title', 'sentiment_compound', 'sentiment_label']].head(3).to_string())
        else:
            print("[X] No sentiment news matched 'Reliance' keyword or feed is empty.")
    except Exception as e:
        print(f"[X] Error fetching sentiment: {e}")
        
    print("\n" + "=" * 60)
    print("Verification execution complete. Review log/errors above.")
    print("=" * 60)

if __name__ == "__main__":
    verify()
