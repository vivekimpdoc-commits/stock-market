import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional

# Import local pipeline modules
from fetch_prices import fetch_historical_daily, clean_ticker_name, fetch_live_indices_data, fetch_live_quote
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

@app.get("/login", response_class=HTMLResponse)
def read_login():
    """
    Serves the new login page.
    """
    login_path = "login.html"
    if os.path.exists(login_path):
        with open(login_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>login.html file not found!</h1>", status_code=404)

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
            "MPT portfolio optimization (/portfolio/optimize)",
            "NSE stock list suggestions (/api/stocks)"
        ]
    }

@app.get("/api/stocks")
def get_stocks():
    """
    Returns a list of all NSE-listed stocks.
    Tries to download the official list from archives.nseindia.com,
    parses it, and caches it locally under data/nse_stocks.json.
    If it fails, falls back to a pre-defined list of ~160 popular stocks.
    """
    import json
    cache_path = os.path.join("data", "nse_stocks.json")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[*] Cache read failed: {e}")
            
    # Try fetching from NSE archives
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    stocks = []
    try:
        import requests
        import csv
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            lines = response.text.splitlines()
            reader = csv.DictReader(lines)
            for row in reader:
                # Filter strictly for EQ (Equity) series to keep standard listings
                series = row.get("SERIES", "").strip().upper()
                if series == "EQ":
                    symbol = row.get("SYMBOL", "").strip().upper()
                    name = row.get("NAME OF COMPANY", "").strip()
                    if symbol and name:
                        stocks.append({"symbol": symbol, "name": name})
                        
            if stocks:
                # Sort alphabetically by symbol
                stocks.sort(key=lambda x: x["symbol"])
                # Save to cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(stocks, f, indent=2, ensure_ascii=False)
                return stocks
    except Exception as e:
        print(f"[-] Failed to fetch official equity list from NSE: {e}")
        
    # Fallback list if fetching fails
    fallback_stocks = [
        {"symbol": "RELIANCE", "name": "Reliance Industries Limited"},
        {"symbol": "TCS", "name": "Tata Consultancy Services Limited"},
        {"symbol": "INFY", "name": "Infosys Limited"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank Limited"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank Limited"},
        {"symbol": "SBIN", "name": "State Bank of India"},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel Limited"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Limited"},
        {"symbol": "ITC", "name": "ITC Limited"},
        {"symbol": "LT", "name": "Larsen & Toubro Limited"},
        {"symbol": "LTIM", "name": "LTIMindtree Limited"},
        {"symbol": "AXISBANK", "name": "Axis Bank Limited"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Limited"},
        {"symbol": "MARUTI", "name": "Maruti Suzuki India Limited"},
        {"symbol": "TATAMOTORS", "name": "Tata Motors Limited"},
        {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Limited"},
        {"symbol": "ONGC", "name": "Oil & Natural Gas Corporation Limited"},
        {"symbol": "COALINDIA", "name": "Coal India Limited"},
        {"symbol": "NTPC", "name": "NTPC Limited"},
        {"symbol": "POWERGRID", "name": "Power Grid Corporation of India Limited"},
        {"symbol": "JSWSTEEL", "name": "JSW Steel Limited"},
        {"symbol": "TATASTEEL", "name": "Tata Steel Limited"},
        {"symbol": "ADANIENT", "name": "Adani Enterprises Limited"},
        {"symbol": "ADANIPORTS", "name": "Adani Ports and Special Economic Zone Limited"},
        {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Limited"},
        {"symbol": "GRASIM", "name": "Grasim Industries Limited"},
        {"symbol": "TECHM", "name": "Tech Mahindra Limited"},
        {"symbol": "CIPLA", "name": "Cipla Limited"},
        {"symbol": "WIPRO", "name": "Wipro Limited"},
        {"symbol": "NESTLEIND", "name": "Nestle India Limited"},
        {"symbol": "BRITANNIA", "name": "Britannia Industries Limited"},
        {"symbol": "ASIANPAINT", "name": "Asian Paints Limited"},
        {"symbol": "APOLLOHOSP", "name": "Apollo Hospitals Enterprise Limited"},
        {"symbol": "DIVISLAB", "name": "Divi's Laboratories Limited"},
        {"symbol": "EICHERMOT", "name": "Eicher Motors Limited"},
        {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp Limited"},
        {"symbol": "INDUSINDBK", "name": "IndusInd Bank Limited"},
        {"symbol": "M&M", "name": "Mahindra & Mahindra Limited"},
        {"symbol": "BPCL", "name": "Bharat Petroleum Corporation Limited"},
        {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Limited"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance Limited"},
        {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv Limited"},
        {"symbol": "TATACONSUM", "name": "Tata Consumer Products Limited"},
        {"symbol": "SBILIFE", "name": "SBI Life Insurance Company Limited"},
        {"symbol": "HDFCLIFE", "name": "HDFC Life Insurance Company Limited"},
        {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories Limited"},
        {"symbol": "TITAN", "name": "Titan Company Limited"},
        {"symbol": "UPL", "name": "UPL Limited"},
        {"symbol": "SHREECEM", "name": "Shree Cement Limited"},
        {"symbol": "HINDALCO", "name": "Hindalco Industries Limited"},
        {"symbol": "DLF", "name": "DLF Limited"},
        {"symbol": "IOC", "name": "Indian Oil Corporation Limited"},
        {"symbol": "HAL", "name": "Hindustan Aeronautics Limited"},
        {"symbol": "BEL", "name": "Bharat Electronics Limited"},
        {"symbol": "IRFC", "name": "Indian Railway Finance Corporation Limited"},
        {"symbol": "PFC", "name": "Power Finance Corporation Limited"},
        {"symbol": "REC", "name": "REC Limited"},
        {"symbol": "GAIL", "name": "GAIL (India) Limited"},
        {"symbol": "SIEMENS", "name": "Siemens Limited"},
        {"symbol": "ABB", "name": "ABB India Limited"},
        {"symbol": "TATAPOWER", "name": "Tata Power Company Limited"},
        {"symbol": "ADANIGREEN", "name": "Adani Green Energy Limited"},
        {"symbol": "ADANIPOWER", "name": "Adani Power Limited"},
        {"symbol": "ATGL", "name": "Adani Total Gas Limited"},
        {"symbol": "AWL", "name": "Adani Wilmar Limited"},
        {"symbol": "UNIONBANK", "name": "Union Bank of India"},
        {"symbol": "CANBK", "name": "Canara Bank"},
        {"symbol": "IDBI", "name": "IDBI Bank Limited"},
        {"symbol": "YESBANK", "name": "Yes Bank Limited"},
        {"symbol": "PNB", "name": "Punjab National Bank"},
        {"symbol": "BANKBARODA", "name": "Bank of Baroda"},
        {"symbol": "IOB", "name": "Indian Overseas Bank"},
        {"symbol": "CENTRALBK", "name": "Central Bank of India"},
        {"symbol": "BOI", "name": "Bank of India"},
        {"symbol": "MAHABANK", "name": "Bank of Maharashtra"},
        {"symbol": "UCOBANK", "name": "UCO Bank"},
        {"symbol": "PSB", "name": "Punjab & Sind Bank"},
        {"symbol": "IDFCFIRSTB", "name": "IDFC First Bank Limited"},
        {"symbol": "FEDERALBNK", "name": "The Federal Bank Limited"},
        {"symbol": "AUBANK", "name": "AU Small Finance Bank Limited"},
        {"symbol": "BANDHANBNK", "name": "Bandhan Bank Limited"},
        {"symbol": "INDHOTEL", "name": "The Indian Hotels Company Limited"},
        {"symbol": "TRENT", "name": "Trent Limited"},
        {"symbol": "NYKAA", "name": "FSN E-Commerce Ventures Limited (Nykaa)"},
        {"symbol": "PAYTM", "name": "One 97 Communications Limited (Paytm)"},
        {"symbol": "ZOMATO", "name": "Zomato Limited"},
        {"symbol": "LIC", "name": "Life Insurance Corporation of India"},
        {"symbol": "GICRE", "name": "General Insurance Corporation of India"},
        {"symbol": "NIACL", "name": "The New India Assurance Company Limited"},
        {"symbol": "JIOFIN", "name": "Jio Financial Services Limited"},
        {"symbol": "POLYCAB", "name": "Polycab India Limited"},
        {"symbol": "KEI", "name": "KEI Industries Limited"},
        {"symbol": "HAVELLS", "name": "Havells India Limited"},
        {"symbol": "VOLTAS", "name": "Voltas Limited"},
        {"symbol": "BLUESTARCO", "name": "Blue Star Limited"},
        {"symbol": "DIXON", "name": "Dixon Technologies (India) Limited"},
        {"symbol": "AMBUJACEM", "name": "Ambuja Cements Limited"},
        {"symbol": "ACC", "name": "ACC Limited"},
        {"symbol": "DALBHARAT", "name": "Dalmia Bharat Limited"},
        {"symbol": "JKCEMENT", "name": "JK Cement Limited"},
        {"symbol": "RAMCOCEM", "name": "The Ramco Cements Limited"},
        {"symbol": "OBEROIRLTY", "name": "Oberoi Realty Limited"},
        {"symbol": "LODHA", "name": "Macrotech Developers Limited (Lodha)"},
        {"symbol": "GODREJPROP", "name": "Godrej Properties Limited"},
        {"symbol": "PRESTIGE", "name": "Prestige Estates Projects Limited"},
        {"symbol": "SOBHA", "name": "Sobha Limited"},
        {"symbol": "PHOENIXLTD", "name": "The Phoenix Mills Limited"},
        {"symbol": "JSWENERGY", "name": "JSW Energy Limited"},
        {"symbol": "NHPC", "name": "NHPC Limited"},
        {"symbol": "SJVN", "name": "SJVN Limited"},
        {"symbol": "IRCTC", "name": "Indian Railway Catering And Tourism Corporation Limited"},
        {"symbol": "CONCOR", "name": "Container Corporation of India Limited"},
        {"symbol": "RVNL", "name": "Rail Vikas Nigam Limited"},
        {"symbol": "IRCON", "name": "Ircon International Limited"},
        {"symbol": "RITES", "name": "Rites Limited"},
        {"symbol": "BHEL", "name": "Bharat Heavy Electricals Limited"},
        {"symbol": "BDL", "name": "Bharat Dynamics Limited"},
        {"symbol": "COCHINSHIP", "name": "Cochin Shipyard Limited"},
        {"symbol": "GRSE", "name": "Garden Reach Shipbuilders & Engineers Limited"},
        {"symbol": "MAZDOCK", "name": "Mazagon Dock Shipbuilders Limited"},
        {"symbol": "HINDZINC", "name": "Hindustan Zinc Limited"},
        {"symbol": "VEDL", "name": "Vedanta Limited"},
        {"symbol": "NMDC", "name": "NMDC Limited"},
        {"symbol": "SAIL", "name": "Steel Authority of India Limited"},
        {"symbol": "HINDCOPPER", "name": "Hindustan Copper Limited"},
        {"symbol": "NATIONALUM", "name": "National Aluminium Company Limited"},
        {"symbol": "TATACOMM", "name": "Tata Communications Limited"},
        {"symbol": "HFCL", "name": "HFCL Limited"},
        {"symbol": "ITI", "name": "ITI Limited"},
        {"symbol": "IDEA", "name": "Vodafone Idea Limited"},
        {"symbol": "INDUSTOWER", "name": "Indus Towers Limited"},
        {"symbol": "GMRINFRA", "name": "GMR Airports Infrastructure Limited"},
        {"symbol": "IRB", "name": "IRB Infrastructure Developers Limited"},
        {"symbol": "KEC", "name": "KEC International Limited"},
        {"symbol": "LTTS", "name": "L&T Technology Services Limited"},
        {"symbol": "KPITTECH", "name": "KPIT Technologies Limited"},
        {"symbol": "PERSISTENT", "name": "Persistent Systems Limited"},
        {"symbol": "COFORGE", "name": "Coforge Limited"},
        {"symbol": "MPHASIS", "name": "Mphasis Limited"},
        {"symbol": "CYIENT", "name": "Cyient Limited"},
        {"symbol": "TATAELXSI", "name": "Tata Elxsi Limited"},
        {"symbol": "AFFLE", "name": "Affle (India) Limited"},
        {"symbol": "JINDALSTEL", "name": "Jindal Steel & Power Limited"},
        {"symbol": "ASTRAL", "name": "Astral Limited"},
        {"symbol": "SUPREMEIND", "name": "Supreme Industries Limited"},
        {"symbol": "FINPIPE", "name": "Finolex Industries Limited"},
        {"symbol": "PRINCEPIPE", "name": "Prince Pipes and Fittings Limited"},
        {"symbol": "ASHOKLEY", "name": "Ashok Leyland Limited"},
        {"symbol": "TVSMOTOR", "name": "TVS Motor Company Limited"},
        {"symbol": "BALKRISIND", "name": "Balkrishna Industries Limited"},
        {"symbol": "MRF", "name": "MRF Limited"},
        {"symbol": "APOLLOTYRE", "name": "Apollo Tyres Limited"},
        {"symbol": "CEAT", "name": "CEAT Limited"},
        {"symbol": "JKTYRE", "name": "JK Tyre & Industries Limited"},
        {"symbol": "EXIDEIND", "name": "Exide Industries Limited"},
        {"symbol": "AMARAJABAT", "name": "Amara Raja Energy & Mobility Limited"},
        {"symbol": "SONACOMS", "name": "Sona BLW Precision Forgings Limited (Sona Comstar)"},
        {"symbol": "BOSCHLTD", "name": "Bosch Limited"},
        {"symbol": "UNOSTRUCT", "name": "Uno Minda Limited"},
        {"symbol": "MOTHERSON", "name": "Samvardhana Motherson International Limited"},
        {"symbol": "BHARATFORG", "name": "Bharat Forge Limited"},
        {"symbol": "CUMMINSIND", "name": "Cummins India Limited"},
        {"symbol": "CARBORUN", "name": "Carborundum Universal Limited"},
        {"symbol": "TIMKEN", "name": "Timken India Limited"},
        {"symbol": "SKFINDIA", "name": "SKF India Limited"},
        {"symbol": "SCHAEFFLER", "name": "Schaeffler India Limited"},
        {"symbol": "THERMAX", "name": "Thermax Limited"},
        {"symbol": "TRIVENI", "name": "Triveni Turbine Limited"},
        {"symbol": "HONAUT", "name": "Honeywell Automation India Limited"},
        {"symbol": "CGPOWER", "name": "CG Power and Industrial Solutions Limited"},
        {"symbol": "L&TFH", "name": "L&T Finance Holdings Limited"},
        {"symbol": "MUTHOOTFIN", "name": "Muthoot Finance Limited"},
        {"symbol": "MANAPPURAM", "name": "Manappuram Finance Limited"},
        {"symbol": "CHOLAFIN", "name": "Cholamandalam Investment and Finance Company Limited"},
        {"symbol": "SHRIRAMFIN", "name": "Shriram Finance Limited"},
        {"symbol": "M&MFIN", "name": "Mahindra & Mahindra Financial Services Limited"},
        {"symbol": "LICHSGFIN", "name": "LIC Housing Finance Limited"},
        {"symbol": "HUDCO", "name": "Housing & Urban Development Corporation Limited"},
        {"symbol": "IREDA", "name": "Indian Renewable Energy Development Agency Limited"},
        {"symbol": "ANGELONE", "name": "Angel One Limited"},
        {"symbol": "CDSL", "name": "Central Depository Services (India) Limited"},
        {"symbol": "MCX", "name": "Multi Commodity Exchange of India Limited"},
        {"symbol": "BSE", "name": "BSE Limited"},
        {"symbol": "IEX", "name": "Indian Energy Exchange Limited"},
        {"symbol": "KFINTECH", "name": "KFin Technologies Limited"},
        {"symbol": "CAMS", "name": "Computer Age Management Services Limited"},
        {"symbol": "HDFCAMC", "name": "HDFC Asset Management Company Limited"},
        {"symbol": "NAM-INDIA", "name": "Nippon Life India Asset Management Limited"},
        {"symbol": "UTIAMC", "name": "UTI Asset Management Company Limited"},
        {"symbol": "SBICARD", "name": "SBI Cards and Payment Services Limited"},
        {"symbol": "MAPMYINDIA", "name": "C.E. Info Systems Limited (MapmyIndia)"},
        {"symbol": "RATEGAIN", "name": "RateGain Travel Technologies Limited"},
        {"symbol": "DELHIVERY", "name": "Delhivery Limited"},
        {"symbol": "EASEMYTRIP", "name": "Easy Trip Planners Limited (EaseMyTrip)"},
        {"symbol": "BLUEDART", "name": "Blue Dart Express Limited"},
        {"symbol": "TCIEXPRESS", "name": "TCI Express Limited"},
        {"symbol": "ALLCARGO", "name": "Allcargo Logistics Limited"},
        {"symbol": "VRLADV", "name": "VRL Logistics Limited"},
        {"symbol": "GPPL", "name": "Gujarat Pipavav Port Limited"},
        {"symbol": "SCI", "name": "Shipping Corporation of India Limited"},
        {"symbol": "GEESHIP", "name": "The Great Eastern Shipping Company Limited"},
        {"symbol": "JSWHL", "name": "JSW Holdings Limited"},
        {"symbol": "TATAINVEST", "name": "Tata Investment Corporation Limited"},
        {"symbol": "PILANIINVS", "name": "Pilani Investment and Industries Corporation Limited"},
        {"symbol": "MANKIND", "name": "Mankind Pharma Limited"},
        {"symbol": "LUPIN", "name": "Lupin Limited"},
        {"symbol": "BIOCON", "name": "Biocon Limited"},
        {"symbol": "GLAND", "name": "Gland Pharma Limited"},
        {"symbol": "SYNGENE", "name": "Syngene International Limited"},
        {"symbol": "LAURUSLABS", "name": "Laurus Labs Limited"},
        {"symbol": "GRANULES", "name": "Granules India Limited"},
        {"symbol": "IPCALAB", "name": "Ipca Laboratories Limited"},
        {"symbol": "ZYDUSLIFE", "name": "Zydus Lifesciences Limited"},
        {"symbol": "TORNTPHARM", "name": "Torrent Pharmaceuticals Limited"},
        {"symbol": "ALKEM", "name": "Alkem Laboratories Limited"},
        {"symbol": "GLAXO", "name": "GlaxoSmithKline Pharmaceuticals Limited"},
        {"symbol": "SANOFI", "name": "Sanofi India Limited"},
        {"symbol": "ABBOTT", "name": "Abbott India Limited"},
        {"symbol": "PIDILITE", "name": "Pidilite Industries Limited"},
        {"symbol": "SRF", "name": "SRF Limited"},
        {"symbol": "DEEPAKNTR", "name": "Deepak Nitrite Limited"},
        {"symbol": "TATACHEM", "name": "Tata Chemicals Limited"},
        {"symbol": "GUJGASLTD", "name": "Gujarat Gas Limited"},
        {"symbol": "IGL", "name": "Indraprastha Gas Limited"},
        {"symbol": "MGL", "name": "Mahanagar Gas Limited"},
        {"symbol": "PETRONET", "name": "Petronet LNG Limited"},
        {"symbol": "OIL", "name": "Oil India Limited"},
        {"symbol": "HPCL", "name": "Hindustan Petroleum Corporation Limited"},
        {"symbol": "MRPL", "name": "Mangalore Refinery & Petrochemicals Limited"},
        {"symbol": "CHENNPETRO", "name": "Chennai Petroleum Corporation Limited"},
        {"symbol": "CESC", "name": "CESC Limited"},
        {"symbol": "TORNTPOWER", "name": "Torrent Power Limited"},
        {"symbol": "SUZLON", "name": "Suzlon Energy Limited"},
        {"symbol": "RPOWER", "name": "Reliance Power Limited"},
        {"symbol": "JPPOWER", "name": "Jaiprakash Power Ventures Limited"}
    ]
    
    # Save the fallback list to cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(fallback_stocks, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[-] Failed to cache fallback equity list: {e}")
        
    return fallback_stocks

@app.get("/api/live/indices")
def get_live_indices():
    """
    Returns live market index data (Nifty 50, Nifty Bank, Nifty Next 50).
    """
    try:
        data = fetch_live_indices_data()
        if not data:
            raise HTTPException(status_code=500, detail="Failed to fetch live indices data.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live/quote/{ticker}")
def get_live_quote(ticker: str):
    """
    Returns live stock price data for the specified ticker.
    """
    try:
        data = fetch_live_quote(ticker)
        if not data:
            raise HTTPException(status_code=404, detail=f"Live quote not found for ticker: {ticker}")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fundamentals/{ticker}")
def get_fundamentals(ticker: str):
    """
    Downloads latest company fundamentals and valuation metrics.
    """
    normalized = normalize_ticker(ticker)
    try:
        df = fetch_key_valuation_metrics(normalized)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"Valuation metrics not found for ticker: {ticker}")
        
        row_dict = df.iloc[0].to_dict()
        return row_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # Check if we should use custom ML model or fallback Consensus model
    has_custom_model = (os.path.exists(xgb_model_path) or os.path.exists(rf_model_path)) and os.path.exists(scaler_path)
    
    # 1. Download the latest price data to build features
    # Fetch last 300 days of data to satisfy indicator calculation (requires at least 200 days for SMA 200)
    logging_df = fetch_historical_daily(normalized, start_date="2023-01-01")
    if logging_df is None or logging_df.empty:
        raise HTTPException(status_code=400, detail=f"Could not download price data for ticker {normalized}")
        
    # 2. Calculate Technical Indicators
    price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
    indicators_df = calculate_technical_indicators(price_file)
    if indicators_df is None or indicators_df.empty:
        raise HTTPException(status_code=500, detail="Failed to calculate technical indicators.")
        
    # 3. Extract the most recent row of features
    latest_row = indicators_df.iloc[-1].copy()
    latest_date = latest_row['Date']
    latest_close = latest_row['Close']
    
    if has_custom_model:
        # Load the model
        model = None
        is_xgb = False
        if os.path.exists(xgb_model_path):
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(xgb_model_path)
                is_xgb = True
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading XGBoost model: {e}")
        else:
            try:
                with open(rf_model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading RandomForest model: {e}")
                
        # Load the scaler
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading scaler: {e}")
            
        # Drop Date to match model inputs
        feature_row_df = latest_row.to_frame().transpose().drop(columns=['Date'])
        feature_names = scaler.feature_names_in_
        try:
            feature_row_aligned = feature_row_df[feature_names].values
        except KeyError as e:
            raise HTTPException(status_code=500, detail=f"Features mismatch. Missing feature from scaler: {e}")
            
        # Scale features and Run Predict
        try:
            scaled_features = scaler.transform(feature_row_aligned)
            prediction = int(model.predict(scaled_features)[0])
            
            try:
                probabilities = model.predict_proba(scaled_features)[0]
                confidence = float(probabilities[prediction])
            except AttributeError:
                confidence = 1.0
                
            recommendation = "BUY" if prediction == 1 else "SELL/HOLD"
            details = (
                "XGBoost AI Model predicts the price trend to be UP tomorrow. Consider buying."
                if prediction == 1 else
                "XGBoost AI Model predicts the price trend to be DOWN or Sideways tomorrow. Consider selling or holding."
            )
            model_type = "XGBoost AI Model"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    else:
        # Fallback Heuristic Consensus Model (MITH)
        try:
            score = 0
            
            close = float(latest_row.get('Close', 0))
            sma50 = float(latest_row.get('SMA_50', 0))
            sma200 = float(latest_row.get('SMA_200', 0))
            ema20 = float(latest_row.get('EMA_20', 0))
            rsi14 = float(latest_row.get('RSI_14', 50))
            macdh = float(latest_row.get('MACDh_12_26_9', 0))
            
            # Check Golden Cross / Trend alignment
            if sma50 > sma200 and close > sma50:
                score += 1.5
            elif sma50 < sma200 and close < sma50:
                score -= 1.5
                
            # Check short term EMA crossover
            if close > ema20:
                score += 1.0
            else:
                score -= 1.0
                
            # Check RSI momentum
            if rsi14 > 55:
                score += 1.0
            elif rsi14 < 45:
                score -= 1.0
                
            # Check MACD histogram
            if macdh > 0:
                score += 1.0
            else:
                score -= 1.0
                
            # Check overall consensus
            if score > 0.5:
                prediction = 1
                recommendation = "BUY"
            elif score < -0.5:
                prediction = 0
                recommendation = "SELL/HOLD"
            else:
                prediction = 0
                recommendation = "HOLD"
                
            # Confidence score based on indicator consensus alignment strength
            confidence = min(max(abs(score) / 4.5, 0.5), 1.0)
            
            # Construct readable details detailing the consensus
            bullish_indicators = []
            bearish_indicators = []
            
            if close > sma50: bullish_indicators.append("Price > SMA50")
            else: bearish_indicators.append("Price < SMA50")
            
            if sma50 > sma200: bullish_indicators.append("SMA50 > SMA200 (Golden Cross)")
            else: bearish_indicators.append("SMA50 < SMA200 (Death Cross)")
            
            if close > ema20: bullish_indicators.append("Price > EMA20")
            else: bearish_indicators.append("Price < EMA20")
            
            if rsi14 > 50: bullish_indicators.append(f"RSI {rsi14:.1f} (Bullish)")
            else: bearish_indicators.append(f"RSI {rsi14:.1f} (Bearish)")
            
            if macdh > 0: bullish_indicators.append("MACD Histogram Positive")
            else: bearish_indicators.append("MACD Histogram Negative")
            
            if prediction == 1:
                details = f"Consensus Trend Model predicts UP trend based on bullish technical indicators: {', '.join(bullish_indicators[:3])}."
            else:
                details = f"Consensus Trend Model predicts DOWN/Sideways trend based on bearish technical indicators: {', '.join(bearish_indicators[:3])}."
                
            model_type = "Consensus Trend Model"
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Consensus prediction error: {e}")
            
    return {
        "ticker": normalized,
        "clean_name": clean_name,
        "prediction_date": latest_date,
        "last_close_price": float(latest_close),
        "prediction_code": prediction,
        "prediction_trend": "UP" if prediction == 1 else "DOWN",
        "confidence_score": confidence,
        "recommendation": recommendation,
        "details": details,
        "model_type": model_type
    }

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
