import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from fetch_prices import fetch_historical_daily, fetch_intraday, clean_ticker_name
from fetch_fundamentals import fetch_financial_sheets, fetch_key_valuation_metrics
from fetch_sentiment import fetch_and_analyze_sentiment
from indicators import calculate_technical_indicators
from data_prep import prepare_ml_data
from model_xgboost import train_xgboost_model
from backtest import run_backtest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_end_to_end_pipeline(ticker: str, start_date: str = "2011-01-01", initial_capital: float = 100000.0) -> bool:
    """
    Runs the entire stock market AI pipeline end-to-end for a given ticker:
    1. Fetch historical price data.
    2. Fetch company fundamentals.
    3. Fetch news headlines & run sentiment analysis.
    4. Calculate technical indicators.
    5. Clean, split, and scale data for ML.
    6. Train the XGBoost trend forecasting classifier.
    7. Run AI strategy backtesting.
    """
    ticker = ticker.strip().upper()
    if not ticker.startswith("^") and "." not in ticker:
        ticker = f"{ticker}.NS"
        
    clean_name = clean_ticker_name(ticker)
    
    print("=" * 70)
    print(f"      RUNNING END-TO-END STOCK MARKET AI PIPELINE FOR: {ticker}      ")
    print("=" * 70)
    
    # Step 1: Download stock daily prices & intraday prices
    print("\n[Step 1/7] Fetching historical and intraday stock prices...")
    price_df = fetch_historical_daily(ticker, start_date=start_date)
    if price_df is None or price_df.empty:
        print("[X] Failed in Step 1: Price download failed.")
        return False
    fetch_intraday(ticker, interval="5m", period="5d")
    
    # Step 2: Download company financials
    print("\n[Step 2/7] Fetching company fundamentals (Balance Sheet, P&L)...")
    fetch_financial_sheets(ticker)
    fetch_key_valuation_metrics(ticker)
    
    # Step 3: Fetch news headlines & sentiment
    print("\n[Step 3/7] Fetching latest live news headlines & analyzing sentiment...")
    fetch_and_analyze_sentiment(keyword_filter=clean_name)
    
    # Step 4: Calculate Technical Indicators
    print("\n[Step 4/7] Calculating technical indicators (RSI, MACD, Moving Averages)...")
    price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
    ind_df = calculate_technical_indicators(price_file)
    if ind_df is None or ind_df.empty:
        print("[X] Failed in Step 4: Technical indicators calculation failed.")
        return False
        
    # Step 5: Prepare ML Datasets (Clean, Split, Scale)
    print("\n[Step 5/7] Preparing and scaling datasets for Machine Learning...")
    ind_file = os.path.join("data", f"{clean_name}_indicators.csv")
    ml_files = prepare_ml_data(ind_file, split_ratio=0.80)
    if not ml_files:
        print("[X] Failed in Step 5: ML data prep failed.")
        return False
        
    # Step 6: Train XGBoost classifier
    print("\n[Step 6/7] Training XGBoost trend classifier model...")
    success = train_xgboost_model(clean_name)
    if not success:
        print("[X] Failed in Step 6: Model training failed.")
        return False
        
    # Step 7: Run Backtesting Simulation
    print("\n[Step 7/7] Executing AI strategy backtesting simulation...")
    backtest_metrics = run_backtest(clean_name, initial_cash=initial_capital)
    if not backtest_metrics:
        print("[X] Failed in Step 7: Strategy backtesting failed.")
        return False
        
    print("\n" + "=" * 70)
    print("      [OK] END-TO-END AI PIPELINE SUCCESSFULLY RUN & COMPLETED!      ")
    print("=" * 70)
    print(f"Results generated for {ticker}:")
    print(f"   - Daily Price Rows:      {len(price_df)}")
    print(f"   - Indicators Calculated: {len(ind_df.columns) - 1}")
    print(f"   - Model Saved:           models/{clean_name}_xgboost_model.json")
    print(f"   - Scaler Saved:          models/{clean_name}_scaler.pkl")
    print(f"   - Total Trades Logged:   {backtest_metrics['Total_Trades']}")
    print(f"   - AI Strategy Return:    {backtest_metrics['AI_Total_Return']:.2%}")
    print(f"   - Benchmark Return:      {backtest_metrics['Benchmark_Total_Return']:.2%}")
    print(f"   - Equity curve plot:     plots/{clean_name}_backtest.png")
    print("=" * 70 + "\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker_arg = sys.argv[1]
    else:
        ticker_arg = input("Enter Stock Ticker to run end-to-end pipeline (e.g. RELIANCE, TCS): ").strip()
        
    if ticker_arg:
        run_end_to_end_pipeline(ticker_arg)
    else:
        print("[!] No ticker provided.")
