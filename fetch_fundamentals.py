import os
import pandas as pd
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_ticker_name(ticker: str) -> str:
    """
    Cleans ticker name for filename usage (e.g., RELIANCE.NS -> RELIANCE).
    """
    return ticker.replace("^", "").replace(".NS", "").replace(".BO", "")

def fetch_financial_sheets(ticker: str, save_dir: str = "data") -> dict:
    """
    Fetches balance sheet, income statement (P&L), and cash flow statement for a company.
    Supports both Annual and Quarterly data.
    
    Parameters:
    - ticker: Ticker symbol (e.g. 'RELIANCE.NS')
    - save_dir: Directory where files will be stored
    
    Returns:
    - Dict with filepaths of saved CSVs
    """
    logging.info(f"Fetching financial statements (Balance Sheet, P&L, Cash Flow) for {ticker}...")
    stock = yf.Ticker(ticker)
    clean_name = clean_ticker_name(ticker)
    os.makedirs(save_dir, exist_ok=True)
    
    saved_files = {}
    
    # List of financial sheets to pull and their corresponding yfinance attributes
    sheets = {
        "balance_sheet_annual": "balance_sheet",
        "balance_sheet_quarterly": "quarterly_balance_sheet",
        "income_statement_annual": "financials",
        "income_statement_quarterly": "quarterly_financials",
        "cashflow_annual": "cashflow",
        "cashflow_quarterly": "quarterly_cashflow"
    }
    
    for sheet_name, attr in sheets.items():
        try:
            df = getattr(stock, attr)
            if df is not None and not df.empty:
                # Format sheet: Index (Metrics) as a column, Transpose so that dates are rows (standard tabular format)
                df_transposed = df.transpose()
                df_transposed = df_transposed.reset_index().rename(columns={'index': 'Date'})
                
                # Format date to YYYY-MM-DD safely
                dates_converted = pd.to_datetime(df_transposed['Date'])
                try:
                    if dates_converted.dt.tz is not None:
                        dates_converted = dates_converted.dt.tz_localize(None)
                except AttributeError:
                    pass
                df_transposed['Date'] = dates_converted.dt.strftime('%Y-%m-%d')
                
                file_path = os.path.join(save_dir, f"{clean_name}_{sheet_name}.csv")
                df_transposed.to_csv(file_path, index=False)
                saved_files[sheet_name] = file_path
                logging.info(f"Saved {sheet_name} to {file_path}")
            else:
                logging.warning(f"No data available for {sheet_name} of {ticker}")
        except Exception as e:
            logging.error(f"Error fetching {sheet_name} for {ticker}: {e}")
            
    return saved_files

def fetch_key_valuation_metrics(ticker: str, save_dir: str = "data") -> pd.DataFrame:
    """
    Fetches key metrics such as PE Ratio, PB Ratio, Market Cap, Beta, Div Yield, etc.
    from ticker.info and structures it.
    
    Parameters:
    - ticker: Ticker symbol (e.g. 'RELIANCE.NS')
    - save_dir: Directory where data will be stored
    
    Returns:
    - DataFrame of key metrics
    """
    logging.info(f"Fetching key valuation metrics for {ticker}...")
    stock = yf.Ticker(ticker)
    clean_name = clean_ticker_name(ticker)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        info = stock.info
        if not info:
            logging.warning(f"No info data retrieved for {ticker}.")
            return None
            
        # Select key metrics of interest for alternative data
        key_fields = {
            "Symbol": info.get("symbol"),
            "Long Name": info.get("longName"),
            "Market Cap (INR)": info.get("marketCap"),
            "Trailing PE": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "Price to Book": info.get("priceToBook"),
            "Beta": info.get("beta"),
            "Trailing EPS": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "Profit Margin": info.get("profitMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "Revenue Growth (YoY)": info.get("revenueGrowth"),
            "Earnings Growth (YoY)": info.get("earningsGrowth"),
            "Total Debt (INR)": info.get("totalDebt"),
            "Total Revenue (INR)": info.get("totalRevenue"),
            "Enterprise Value (INR)": info.get("enterpriseValue"),
            "Book Value": info.get("bookValue"),
            "Free Cash Flow": info.get("freeCashflow")
        }
        
        # Filter out fields that are None
        filtered_fields = {k: [v] for k, v in key_fields.items() if v is not None}
        
        if not filtered_fields:
            logging.warning(f"No valid key metrics found for {ticker}")
            return None
            
        df = pd.DataFrame(filtered_fields)
        file_path = os.path.join(save_dir, f"{clean_name}_key_metrics.csv")
        df.to_csv(file_path, index=False)
        logging.info(f"Saved key valuation metrics to {file_path}")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching key valuation metrics for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Test script locally
    test_ticker = "RELIANCE.NS"
    print(f"--- Fetching Financial Sheets for {test_ticker} ---")
    fetch_financial_sheets(test_ticker)
    
    print(f"\n--- Fetching Key Metrics for {test_ticker} ---")
    fetch_key_valuation_metrics(test_ticker)
