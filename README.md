# Indian Stock Market Data Acquisition Pipeline
## (भारतीय शेयर बाजार डेटा संग्रहण पाइपलाइन)

This project contains Python scripts to acquire stock prices, company fundamentals, and financial news sentiment analysis for AI model training or analysis.

यह प्रोजेक्ट भारतीय शेयर बाजार (NSE/BSE) के ऐतिहासिक और लाइव डेटा को डाउनलोड करने, मैनेज करने, और समाचारों का सेंटीमेंट एनालिसिस (Sentiment Analysis) करने के लिए बनाया गया है।

---

## 📁 Project Structure (प्रोजेक्ट संरचना)

- `requirements.txt`: Project dependencies (yfinance, pandas, vaderSentiment, requests, beautifulsoup4, lxml).
- `fetch_prices.py`: Fetches historical daily and intraday prices from Yahoo Finance.
- `fetch_fundamentals.py`: Fetches Balance Sheet, Income Statement (P&L), and key metrics like PE ratio, PB, etc.
- `fetch_sentiment.py`: Scrapes financial news headlines (Moneycontrol, Economic Times, Livemint) via RSS and computes VADER sentiment scores.
- `main.py`: Interactive CLI tool to run specific modules or the whole pipeline for any stock.
- `data/`: Folder generated automatically containing the saved CSV data.

---

## 🛠 Setup & Installation (सेटअप और इंस्टॉलेशन)

1. Make sure you have **Python 3.8+** installed. (सुनिश्चित करें कि आपके पास Python 3.8 या उससे नया वर्शन इंस्टॉल है।)
2. Open terminal in this folder and install dependencies:
   (इस फ़ोल्डर में टर्मिनल खोलें और निम्नलिखित कमांड चलाकर लाइब्रेरी इंस्टॉल करें:)
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run (कैसे चलाएं)

Run the interactive CLI manager:
(इंटरैक्टिव कमांड-लाइन टूल चलाने के लिए यह कमांड रन करें:)

```bash
python main.py
```

### Options (विकल्प):
1. **Option 1**: Download 15-year daily prices (e.g. `RELIANCE.NS`, `TCS.NS` or index `^NSEI`) and intraday 5-min intervals.
2. **Option 2**: Download Financial statements (Balance Sheet, Quarterly Income Statements) and key ratios.
3. **Option 3**: Fetch latest business headlines and calculate positive, negative, and compound sentiment scores.
4. **Option 4 (Complete Pipeline)**: Automatically fetch prices, financials, and news sentiments for a specific stock in one single command.

---

## 📊 Data Outputs (सेव होने वाला डेटा)

All data is stored inside a `data/` directory in standard CSV format:
- `<TICKER>_daily_prices.csv`: Open, High, Low, Close, Volume for training time-series models.
- `<TICKER>_intraday_5m.csv`: Intraday prices for high-frequency strategies.
- `<TICKER>_balance_sheet_annual.csv` / `_quarterly.csv`: Fundamental balance sheet values.
- `<TICKER>_income_statement_annual.csv` / `_quarterly.csv`: Revenue, net profit, and general P&L statements.
- `<TICKER>_key_metrics.csv`: Snapshot of current valuation (PE Ratio, Price to Book, Dividend Yield, EPS, etc.).
- `market_sentiment.csv`: Latest headlines, source links, dates, and calculated sentiment scores.
