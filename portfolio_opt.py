import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from fetch_prices import fetch_historical_daily, clean_ticker_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optimize_portfolio(tickers: list, start_date: str = "2020-01-01", num_portfolios: int = 5000, risk_free_rate: float = 0.06) -> tuple:
    """
    Downloads historical price data for a list of stock tickers, merges closing prices,
    performs Markowitz Mean-Variance optimization via Monte Carlo simulations,
    finds the Max Sharpe Ratio & Min Volatility portfolios, saves weights, and plots the Efficient Frontier.
    
    Parameters:
    - tickers: List of ticker strings (e.g. ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS'])
    - start_date: Start date string (default: 2020-01-01 for 6+ years of data)
    - num_portfolios: Number of simulated portfolios (default: 5000)
    - risk_free_rate: Annual risk-free rate of return (default: 0.06 or 6% for Indian markets)
    
    Returns:
    - Tuple (max_sharpe_portfolio_dict, min_vol_portfolio_dict)
    """
    if len(tickers) < 2:
        logging.error("Portfolio optimization requires at least 2 tickers.")
        return None
        
    logging.info(f"Starting portfolio optimization for tickers: {tickers}...")
    
    # 1. Download price data for all tickers and extract close prices
    close_prices = {}
    
    for ticker in tickers:
        clean_name = clean_ticker_name(ticker)
        price_file = os.path.join("data", f"{clean_name}_daily_prices.csv")
        
        # Download if not locally available
        if not os.path.exists(price_file):
            logging.info(f"Price file for {ticker} not found locally. Downloading...")
            df = fetch_historical_daily(ticker, start_date=start_date)
        else:
            df = pd.read_csv(price_file)
            
        if df is not None and not df.empty and 'Close' in df.columns:
            # Set Date index for alignment
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            close_prices[clean_name] = df['Close']
        else:
            logging.warning(f"Skipping {ticker}: Could not retrieve closing price.")
            
    if len(close_prices) < 2:
        logging.error("Failed to fetch price data for at least 2 stocks. Optimization aborted.")
        return None
        
    # Combine into a single DataFrame
    portfolio_df = pd.DataFrame(close_prices).dropna()
    logging.info(f"Aligned historical closing prices shape: {portfolio_df.shape} (rows, stocks).")
    
    # Calculate daily returns
    returns = portfolio_df.pct_change().dropna()
    
    # Annualized mean returns and covariance matrix
    # (assuming 252 trading days in a year)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    num_assets = len(close_prices)
    stock_names = list(close_prices.keys())
    
    # Lists to store portfolio stats
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    np.random.seed(42) # For reproducible random portfolios
    
    # 2. Monte Carlo Simulation
    logging.info(f"Simulating {num_portfolios} portfolios...")
    for i in range(num_portfolios):
        # Generate random weights summing to 1.0
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Expected annual portfolio return
        p_return = np.sum(weights * mean_returns)
        # Expected annual portfolio volatility
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Sharpe Ratio (return - risk_free_rate) / volatility
        p_sharpe = (p_return - risk_free_rate) / p_volatility
        
        results[0, i] = p_return
        results[1, i] = p_volatility
        results[2, i] = p_sharpe
        
    # 3. Locate optimal portfolios
    max_sharpe_idx = np.argmax(results[2])
    sd_max_sharpe, r_max_sharpe = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    w_max_sharpe = weights_record[max_sharpe_idx]
    
    min_vol_idx = np.argmin(results[1])
    sd_min_vol, r_min_vol = results[1, min_vol_idx], results[0, min_vol_idx]
    w_min_vol = weights_record[min_vol_idx]
    
    # Print results
    print("\n" + "=" * 60)
    print("        MODERN PORTFOLIO THEORY (MPT) OPTIMIZATION RESULTS")
    print("=" * 60)
    print("MAXIMUM SHARPE RATIO PORTFOLIO (Optimal Returns/Risk Ratio):")
    print(f"   Annualized Expected Return: {r_max_sharpe:.2%}")
    print(f"   Annualized Volatility:      {sd_max_sharpe:.2%}")
    print(f"   Sharpe Ratio:               {results[2, max_sharpe_idx]:.4f}")
    print("   Stock Weights Allocation:")
    for stock, weight in zip(stock_names, w_max_sharpe):
        print(f"      - {stock}: {weight:.2%}")
        
    print("\nMINIMUM VOLATILITY PORTFOLIO (Lowest Risk Allocation):")
    print(f"   Annualized Expected Return: {r_min_vol:.2%}")
    print(f"   Annualized Volatility:      {sd_min_vol:.2%}")
    print(f"   Sharpe Ratio:               {results[2, min_vol_idx]:.4f}")
    print("   Stock Weights Allocation:")
    for stock, weight in zip(stock_names, w_min_vol):
        print(f"      - {stock}: {weight:.2%}")
    print("=" * 60 + "\n")
    
    # 4. Save results to CSV
    os.makedirs("data", exist_ok=True)
    weights_df = pd.DataFrame({
        "Stock": stock_names,
        "Max_Sharpe_Weight": w_max_sharpe,
        "Min_Volatility_Weight": w_min_vol
    })
    weights_file = os.path.join("data", "portfolio_optimal_weights.csv")
    weights_df.to_csv(weights_file, index=False)
    logging.info(f"[✓] Optimal weights saved to {weights_file}")
    
    # 5. Plot the Efficient Frontier
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Scatter plot representing simulated portfolios colored by Sharpe Ratio
    sc = plt.scatter(results[1], results[0], c=results[2], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, label='Sharpe Ratio')
    
    # Plot Max Sharpe Portfolio (Green Star)
    plt.scatter(sd_max_sharpe, r_max_sharpe, marker='*', color='green', s=200, label='Max Sharpe Ratio')
    # Plot Min Volatility Portfolio (Red Star)
    plt.scatter(sd_min_vol, r_min_vol, marker='*', color='red', s=200, label='Min Volatility')
    
    plt.title('Efficient Frontier - Indian Stock Portfolio')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Expected Returns')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_file = os.path.join("plots", "portfolio_efficient_frontier.png")
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"[✓] Efficient Frontier plot saved to {plot_file}")
    
    # Construct outputs
    max_sharpe_portfolio = {
        "return": r_max_sharpe, "volatility": sd_max_sharpe, "sharpe": results[2, max_sharpe_idx],
        "weights": dict(zip(stock_names, w_max_sharpe))
    }
    min_vol_portfolio = {
        "return": r_min_vol, "volatility": sd_min_vol, "sharpe": results[2, min_vol_idx],
        "weights": dict(zip(stock_names, w_min_vol))
    }
    
    return max_sharpe_portfolio, min_vol_portfolio

if __name__ == "__main__":
    # Test locally
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
    optimize_portfolio(test_stocks, start_date="2024-01-01", num_portfolios=1000)
