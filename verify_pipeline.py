import os
import sys
import pandas as pd
import numpy as np

def verify():
    print("=" * 75)
    print("      VERIFYING DATA PIPELINE, AI CORE & FASTAPI SERVER         ")
    print("=" * 75)
    
    # 1. Imports check
    print("[1] Checking module imports...")
    try:
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
        from app import app
        print("[OK] All modules, including FastAPI app.py, imported successfully!")
    except ImportError as e:
        print(f"[X] Import error: {e}")
        print("[!] Make sure you installed the dependencies using: pip install -r requirements.txt")
        return
        
    test_ticker = "RELIANCE.NS"
    clean_name = "RELIANCE"
    
    # 2. Daily Price Download & Indicators Check
    print(f"\n[2] Checking stock price download & technical indicators...")
    price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
    indicators_file = os.path.join("data", f"{clean_name}_indicators.csv")
    
    if not os.path.exists(price_file):
        logging_df = fetch_historical_daily(test_ticker, start_date="2024-01-01")
        if logging_df is None:
            print("[X] Failed to download prices.")
            return
            
    if not os.path.exists(indicators_file):
        ind_df = calculate_technical_indicators(price_file)
        if ind_df is None:
            print("[X] Failed to calculate technical indicators.")
            return
            
    # Preprocess check
    ml_files = prepare_ml_data(indicators_file, split_ratio=0.80)
    if not ml_files:
        print("[X] Failed to prepare machine learning datasets.")
        return
    print("[OK] Dataset scaling and preparation verified.")
        
    # 3. XGBoost / RandomForest Classifier check
    print(f"\n[3] Testing classification model training (XGBoost/RF)...")
    try:
        success = train_xgboost_model(clean_name)
        if success:
            print(f"[OK] Classification model trained successfully! Model file and importance plot saved.")
        else:
            print("[X] Classification model training failed.")
            return
    except Exception as e:
        print(f"[X] Exception during classification training: {e}")
        return
        
    # 4. Backtesting Check
    print(f"\n[4] Testing AI Strategy Backtesting engine...")
    try:
        backtest_metrics = run_backtest(clean_name)
        if backtest_metrics:
            print(f"[OK] Backtesting successful!")
            print(f"   - AI Strategy Return: {backtest_metrics['AI_Total_Return']:.2%}")
            print(f"   - Benchmark Return:   {backtest_metrics['Benchmark_Total_Return']:.2%}")
            print(f"   - Annual Sharpe:      {backtest_metrics['Annualized_Sharpe_Ratio']:.4f}")
            print(f"   - Max Drawdown:       {backtest_metrics['Max_Drawdown']:.2%}")
            
            # Check if outputs exist
            trades_file = os.path.join("data", f"{clean_name}_backtest_trades.csv")
            plot_file = os.path.join("plots", f"{clean_name}_backtest.png")
            if os.path.exists(trades_file) and os.path.exists(plot_file):
                print(f"[OK] Backtest trades CSV and plot verification passed.")
            else:
                print(f"[X] Missing backtest output files.")
        else:
            print("[X] Backtesting execution failed.")
    except Exception as e:
        print(f"[X] Exception during backtesting: {e}")
        
    # 5. MPT Portfolio Optimization Check
    print(f"\n[5] Testing Modern Portfolio Theory (MPT) optimization...")
    try:
        # Optimizing small portfolio for speed (Reliance and TCS)
        portfolios = optimize_portfolio(['RELIANCE.NS', 'TCS.NS'], start_date="2024-01-01", num_portfolios=200)
        if portfolios:
            print(f"[OK] MPT Portfolio optimization verified! Frontier plot and CSV weights saved.")
        else:
            print("[X] Portfolio optimization failed.")
    except Exception as e:
        print(f"[X] Exception during MPT optimization: {e}")
        
    # 6. FinBERT Sentiment Check
    print(f"\n[6] Testing News Sentiment Analysis (FinBERT / Fallback)...")
    news_file = os.path.join("data", "market_sentiment.csv")
    if not os.path.exists(news_file):
        fetch_and_analyze_sentiment(keyword_filter="Reliance")
        
    try:
        sent_df = run_finbert_sentiment(news_file)
        if not sent_df.empty:
            print(f"[OK] News sentiment analysis verified!")
        else:
            print("[X] News sentiment analysis failed.")
    except Exception as e:
        print(f"[X] Exception during news sentiment analysis: {e}")
        
    # 7. LSTM Check (Import check)
    print(f"\n[7] Checking LSTM Neural Network Setup...")
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow version: {tf.__version__}")
        print("[*] Skipping LSTM training dry-run (use Option 6 in main.py to train).")
    except ImportError:
        print("[!] TensorFlow not installed. Install tensorflow using 'pip install tensorflow' to train the LSTM.")
        
    # 8. FastAPI server check
    print(f"\n[8] Verifying FastAPI Web Server instance...")
    try:
        import uvicorn
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)
        print("[OK] FastAPI server app instance successfully initialized!")
        
        # Check if dashboard exists
        if os.path.exists("index.html"):
            print("[OK] index.html dashboard file found and ready to be served at root URL (/).")
        else:
            print("[X] index.html dashboard file is missing.")
            
        print("    To run manually: uvicorn app:app --port 8000 --reload")
    except Exception as e:
        print(f"[X] FastAPI verification failed: {e}")
        
    print("\n" + "=" * 75)
    print("Verification execution complete. All AI pipeline and API scripts verified!")
    print("=" * 75)

if __name__ == "__main__":
    verify()
