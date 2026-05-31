import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional

# Import local pipeline modules
from fetch_prices import fetch_historical_daily, clean_ticker_name
from fetch_fundamentals import fetch_financial_sheets, fetch_key_valuation_metrics
from fetch_sentiment import fetch_and_analyze_sentiment
from indicators import calculate_technical_indicators
from portfolio_opt import optimize_portfolio

app = FastAPI(
    title="Indian Stock Market AI API",
    description="FastAPI microservice for real-time stock price trend prediction, news sentiment scoring, and MPT portfolio optimization.",
    version="1.0.0"
)

# Helper function to convert input ticker to standard NSE format
def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker and not ticker.startswith("^") and "." not in ticker:
        return f"{ticker}.NS"
    return ticker

class PortfolioRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = "2024-01-01"
    risk_free_rate: Optional[float] = 0.06

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_index():
    """
    Serves the interactive glassmorphic stock market dashboard.
    """
    index_path = "index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html dashboard file not found!</h1>", status_code=404)

@app.get("/status")
def status():
    """
    Health-check endpoint showing API status and server datetime.
    """
    return {
        "status": "online",
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "supported_features": [
            "Real-time prediction (/predict/{ticker})",
            "News sentiment analysis (/sentiment/{ticker})",
            "MPT portfolio optimization (/portfolio/optimize)"
        ]
    }

@app.get("/predict/{ticker}")
def predict(ticker: str):
    """
    Downloads latest stock prices, calculates technical indicators, scales features
    using the saved scaler, and runs the classification model to output trading recommendation (BUY/SELL/HOLD).
    """
    normalized = normalize_ticker(ticker)
    clean_name = clean_ticker_name(normalized)
    
    # Check if model and scaler files exist
    models_dir = "models"
    xgb_model_path = os.path.join(models_dir, f"{clean_name}_xgboost_model.json")
    rf_model_path = os.path.join(models_dir, f"{clean_name}_randomforest_model.pkl")
    scaler_path = os.path.join(models_dir, f"{clean_name}_scaler.pkl")
    
    model = None
    is_xgb = False
    
    # 1. Load the Model
    if os.path.exists(xgb_model_path):
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(xgb_model_path)
            is_xgb = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading XGBoost model: {e}")
    elif os.path.exists(rf_model_path):
        try:
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading RandomForest model: {e}")
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"Trained model for {clean_name} not found. Please train a classifier model for this ticker using main.py option 7 first."
        )
        
    # 2. Load the Scaler
    if not os.path.exists(scaler_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Scaler for {clean_name} not found. Please run preprocessing first using main.py option 5."
        )
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scaler: {e}")
        
    # 3. Download the latest price data to build features
    # Fetch last 300 days of data to satisfy indicator calculation (requires at least 200 days for SMA 200)
    logging_df = fetch_historical_daily(normalized, start_date="2023-01-01")
    if logging_df is None or logging_df.empty:
        raise HTTPException(status_code=400, detail=f"Could not download price data for ticker {normalized}")
        
    # 4. Calculate Technical Indicators
    price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
    indicators_df = calculate_technical_indicators(price_file)
    if indicators_df is None or indicators_df.empty:
        raise HTTPException(status_code=500, detail="Failed to calculate technical indicators.")
        
    # 5. Extract the most recent row of features
    latest_row = indicators_df.iloc[-1].copy()
    latest_date = latest_row['Date']
    latest_close = latest_row['Close']
    
    # Drop Date to match model inputs
    feature_row_df = latest_row.to_frame().transpose().drop(columns=['Date'])
    
    # Align features list with what scaler was trained on
    # In case there are mismatch columns, we filter
    feature_names = scaler.feature_names_in_
    try:
        feature_row_aligned = feature_row_df[feature_names].values
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Features mismatch. Missing feature from scaler: {e}")
        
    # 6. Scale features and Run Predict
    try:
        scaled_features = scaler.transform(feature_row_aligned)
        prediction = int(model.predict(scaled_features)[0])
        
        # Get probability if available
        try:
            probabilities = model.predict_proba(scaled_features)[0]
            confidence = float(probabilities[prediction])
        except AttributeError:
            confidence = 1.0  # fallback if model doesn't support probability
            
        recommendation = "BUY" if prediction == 1 else "SELL/HOLD"
        details = (
            "Model predicts the price trend to be UP tomorrow. Consider buying."
            if prediction == 1 else
            "Model predicts the price trend to be DOWN or Sideways tomorrow. Consider selling or holding."
        )
        
        return {
            "ticker": normalized,
            "clean_name": clean_name,
            "prediction_date": latest_date,
            "last_close_price": float(latest_close),
            "prediction_code": prediction,
            "prediction_trend": "UP" if prediction == 1 else "DOWN",
            "confidence_score": confidence,
            "recommendation": recommendation,
            "details": details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/sentiment/{ticker}")
def sentiment(ticker: str):
    """
    Parses live financial news feeds and computes sentiment scores for headlines
    mentioning the stock ticker.
    """
    clean_name = clean_ticker_name(normalize_ticker(ticker))
    
    try:
        sentiment_df = fetch_and_analyze_sentiment(keyword_filter=clean_name)
        if sentiment_df.empty:
            return {
                "ticker": ticker,
                "clean_name": clean_name,
                "news_found": 0,
                "sentiment_summary": "Neutral (No articles found)",
                "mean_compound_score": 0.0,
                "headlines": []
            }
            
        mean_score = float(sentiment_df['sentiment_compound'].mean())
        
        if mean_score >= 0.05:
            sentiment_summary = "Positive"
        elif mean_score <= -0.05:
            sentiment_summary = "Negative"
        else:
            sentiment_summary = "Neutral"
            
        # Extract headlines
        headlines_list = []
        for _, row in sentiment_df.head(10).iterrows():
            headlines_list.append({
                "source": row['source'],
                "headline": row['title'],
                "published_at": row['published_at'],
                "score": float(row['sentiment_compound']),
                "label": row['sentiment_label']
            })
            
        return {
            "ticker": ticker,
            "clean_name": clean_name,
            "news_found": len(sentiment_df),
            "sentiment_summary": sentiment_summary,
            "mean_compound_score": mean_score,
            "headlines": headlines_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {e}")

@app.post("/portfolio/optimize")
def optimize(request: PortfolioRequest):
    """
    Accepts a list of tickers and runs Modern Portfolio Theory (MPT) Mean-Variance
    optimization, returning Max Sharpe Ratio and Min Volatility allocations.
    """
    normalized_tickers = [normalize_ticker(t) for t in request.tickers]
    
    try:
        opt_results = optimize_portfolio(
            tickers=normalized_tickers,
            start_date=request.start_date,
            risk_free_rate=request.risk_free_rate,
            num_portfolios=1000  # small limit for API responsiveness
        )
        
        if not opt_results:
            raise HTTPException(status_code=400, detail="Portfolio optimization failed. Check ticker symbols.")
            
        max_sharpe, min_vol = opt_results
        
        return {
            "tickers_analyzed": normalized_tickers,
            "max_sharpe_ratio_portfolio": {
                "expected_return": float(max_sharpe["return"]),
                "expected_volatility": float(max_sharpe["volatility"]),
                "sharpe_ratio": float(max_sharpe["sharpe"]),
                "weights": {k: float(v) for k, v in max_sharpe["weights"].items()}
            },
            "minimum_volatility_portfolio": {
                "expected_return": float(min_vol["return"]),
                "expected_volatility": float(min_vol["volatility"]),
                "sharpe_ratio": float(min_vol["sharpe"]),
                "weights": {k: float(v) for k, v in min_vol["weights"].items()}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization error: {e}")
