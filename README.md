# Indian Stock Market Data Acquisition, AI Core, Backtesting & Web API Pipeline
## (भारतीय शेयर बाजार डेटा संग्रहण, एआई कोर, बैकटेस्टिंग और वेब एपीआई पाइपलाइन)

This project contains Python scripts to acquire stock prices, company fundamentals, calculate technical indicators, train AI models (LSTM, XGBoost), run FinBERT news sentiment analysis, backtest trading strategies, and deploy the AI core via a production FastAPI web server for integration with Flutter, React Native, or web clients.

यह प्रोजेक्ट भारतीय शेयर बाजार (NSE/BSE) के डेटा संग्रहण, तकनीकी संकेतकों की गणना, एआई मॉडल्स की ट्रेनिंग (LSTM, XGBoost), FinBERT समाचार सेंटीमेंट एनालिसिस, एआई ट्रेडिंग बैकटेस्टिंग और मोबाइल/वेब एप्स के एकीकरण हेतु **FastAPI** वेब सर्वर पर आधारित डिप्लॉयमेंट के लिए बनाया गया है।

---

## 📁 Project Structure (प्रोजेक्ट संरचना)

- `requirements.txt`: Project dependencies.
- `fetch_prices.py`: Fetches historical daily and intraday prices from Yahoo Finance.
- `fetch_fundamentals.py`: Fetches Balance Sheet, Income Statement (P&L), and key ratios.
- `fetch_sentiment.py`: Scrapes financial news headlines via RSS and computes VADER sentiment scores.
- `indicators.py`: Calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV) using `pandas-ta`.
- `data_prep.py`: Preprocesses data for ML. Saves fitted `StandardScaler` to `models/{ticker}_scaler.pkl`.
- `model_lstm.py`: Trains a deep learning LSTM model in TensorFlow.
- `model_xgboost.py`: Trains an XGBoost Classifier to predict stock up/down direction.
- `model_finbert.py`: Scores news headlines using the pre-trained Hugging Face **FinBERT** transformer model.
- `portfolio_opt.py`: Calculates optimal asset weight allocations using Modern Portfolio Theory (MPT).
- `backtest.py`: Simulates trading based on AI signals (Buy/Sell) factoring in fees.
- `run_pipeline.py`: Runs all pipeline steps (Download, Indicators, Scale, Train, and Backtest) end-to-end for a stock in one single command.
- `app.py`: Sets up a **FastAPI** web server serving status, predictions, news sentiment, and MPT allocations.
- `main.py`: Interactive CLI console menu to run specific modules or the end-to-end pipeline runner.
- `verify_pipeline.py`: Helper script to automatically test and verify the entire pipeline end-to-end.
- `data/`: Folder containing the saved CSV data.
- `models/`: Folder containing trained and saved model weights.
- `plots/`: Folder containing charts and plots.

---

## 🛠 Setup & Installation (सेटअप और इंस्टॉलेशन)

1. Make sure you have **Python 3.8+** installed. (सुनिश्चित करें कि आपके पास Python 3.8 या उससे नया वर्शन इंस्टॉल है।)
2. Open terminal in this folder and install dependencies:
   (इस फ़ोल्डर में टर्मिनल खोलें और निम्नलिखित कमांड चलाकर लाइब्रेरी इंस्टॉल करें:)
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Deep learning libraries (TensorFlow, PyTorch) are large packages (approx. 1.5GB total download).*

---

## 🚀 How to Run (कैसे चलाएं)

### 1. Run Everything in One Go (एक बार में पूरा पाइपलाइन चलाएं)
You can run all steps end-to-end (Download -> Calculate Indicators -> Scale Data -> Train XGBoost -> Run Backtest) for any stock ticker (e.g., RELIANCE, TCS) using a single command:
(आप किसी भी शेयर (जैसे RELIANCE, TCS) के लिए पूरी पाइपलाइन एक ही बार में चलाने के लिए यह कमांड रन कर सकते हैं:)
```bash
python run_pipeline.py
```
This will download the data, calculate features, train the AI model, simulate historical trades, and output the performance report.

### 2. Interactive CLI Console
Run the interactive CLI manager:
(इंटरैक्टिव कमांड-लाइन टूल चलाने के लिए यह कमांड रन करें:)
```bash
python main.py
```
Select **Option 11** to run the end-to-end pipeline, or **Option 12** to launch the FastAPI server.

### 3. Launch FastAPI Server Directly
Start the FastAPI server using Uvicorn:
(वेब एपीआई होस्टिंग सर्वर शुरू करने के लिए यह कमांड रन करें:)
```bash
python -m uvicorn app:app --port 8000 --reload
```
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to test the API endpoints using the interactive Swagger UI.

---

## ⚡ API Endpoints (एपीआई एंडपॉइंट्स)

- `GET /status`: Health check confirming status.
- `GET /predict/{ticker}`: Feeds latest indicators into the trained model to return recommendation: `BUY` or `SELL/HOLD` with confidence score.
- `GET /sentiment/{ticker}`: Real-time news headlines sentiment positive/negative summary.
- `POST /portfolio/optimize`: Computes optimal portfolio weights.
