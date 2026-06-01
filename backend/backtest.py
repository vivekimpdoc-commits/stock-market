import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_backtest(ticker_prefix: str, initial_cash: float = 100000.0, commission: float = 0.0005) -> dict:
    """
    Loads preprocessed test features and the trained classification model.
    Simulates trading: Buy when model predicts Up (1), Sell when Down (0).
    Compares against a Buy & Hold benchmark, calculates standard metrics (Returns, Max Drawdown, Sharpe),
    saves a trades log CSV, and plots the portfolio equity curves.
    
    Parameters:
    - ticker_prefix: The prefix name of the stock file (e.g., RELIANCE)
    - initial_cash: Starting capital in INR (default: 100,000 INR)
    - commission: Transaction fee rate per trade (default: 0.05% or 0.0005)
    
    Returns:
    - Dict of performance metrics, or None if failed.
    """
    # Define directories relative to the execution root directory
    data_dir = os.path.join("data", "preprocessed")
    models_dir = "models"
    plots_dir = "plots"
    
    # 1. Check if files exist
    x_test_path = os.path.join(data_dir, f"{ticker_prefix}_X_test.csv")
    y_test_path = os.path.join(data_dir, f"{ticker_prefix}_y_test_clf.csv")
    daily_prices_path = os.path.join("data", f"{ticker_prefix}_daily_prices.csv")
    
    for path in [x_test_path, y_test_path, daily_prices_path]:
        if not os.path.exists(path):
            logging.error(f"Required backtesting file '{path}' is missing.")
            return None
            
    # Check for model files
    xgb_model_path = os.path.join(models_dir, f"{ticker_prefix}_xgboost_model.json")
    rf_model_path = os.path.join(models_dir, f"{ticker_prefix}_randomforest_model.pkl")
    
    if os.path.exists(xgb_model_path):
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(xgb_model_path)
            logging.info(f"Loaded trained XGBoost model from {xgb_model_path}")
        except Exception as e:
            logging.error(f"Error loading XGBoost model: {e}")
            return None
    elif os.path.exists(rf_model_path):
        try:
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Loaded trained RandomForest model from {rf_model_path}")
        except Exception as e:
            logging.error(f"Error loading RandomForest model: {e}")
            return None
    else:
        logging.error(f"No trained classification model (XGBoost or RandomForest) found for {ticker_prefix} under models/.")
        return None
        
    # 2. Load and align datasets
    X_test_df = pd.read_csv(x_test_path)
    daily_prices_df = pd.read_csv(daily_prices_path)
    
    # Get feature names (excluding Date)
    feature_names = [col for col in X_test_df.columns if col != 'Date']
    X_features = X_test_df[feature_names].values
    
    # Generate predictions (signals)
    predictions = model.predict(X_features)
    
    # Align actual close prices with test dates
    # Merging ensures dates match perfectly
    test_dates_df = X_test_df[['Date']].copy()
    test_dates_df['Signal'] = predictions
    
    # Merge with daily prices to get actual Close prices
    test_data = pd.merge(test_dates_df, daily_prices_df[['Date', 'Close']], on='Date', how='inner')
    test_data = test_data.sort_values(by='Date').reset_index(drop=True)
    
    if test_data.empty:
        logging.error("Failed to align test dates with historical daily close prices.")
        return None
        
    # 3. Simulate Trading Loop
    cash = initial_cash
    position = 0  # Number of shares held
    portfolio_history = []
    trade_log = []
    
    # Track trade entry to compute win/loss stats
    last_buy_price = 0.0
    profitable_trades = 0
    total_completed_trades = 0
    
    logging.info(f"Running backtest simulation for {len(test_data)} trading days...")
    for idx, row in test_data.iterrows():
        date = row['Date']
        price = row['Close']
        signal = row['Signal']
        
        # Action Logic:
        # Buy Signal (1) and not holding stock -> BUY
        if signal == 1 and position == 0:
            shares_to_buy = int((cash * (1.0 - commission)) // price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                fee = cost * commission
                cash = cash - cost - fee
                position = shares_to_buy
                last_buy_price = price
                
                trade_log.append({
                    "Date": date,
                    "Action": "BUY",
                    "Price": price,
                    "Shares": shares_to_buy,
                    "Transaction_Fee": fee,
                    "Cash": cash,
                    "Portfolio_Value": cash + (position * price)
                })
                
        # Sell Signal (0) and holding stock -> SELL
        elif signal == 0 and position > 0:
            revenue = position * price
            fee = revenue * commission
            cash = cash + revenue - fee
            
            # Check if profitable
            trade_profit = price - last_buy_price
            if trade_profit > 0:
                profitable_trades += 1
            total_completed_trades += 1
            
            trade_log.append({
                "Date": date,
                "Action": "SELL",
                "Price": price,
                "Shares": position,
                "Transaction_Fee": fee,
                "Cash": cash,
                "Portfolio_Value": cash
            })
            position = 0
            
        # Record daily equity value
        current_val = cash + (position * price)
        portfolio_history.append(current_val)
        
    # Close out open position on last day if holding
    if position > 0:
        last_row = test_data.iloc[-1]
        price = last_row['Close']
        revenue = position * price
        fee = revenue * commission
        cash = cash + revenue - fee
        
        trade_profit = price - last_buy_price
        if trade_profit > 0:
            profitable_trades += 1
        total_completed_trades += 1
        
        trade_log.append({
            "Date": last_row['Date'],
            "Action": "SELL_CLOSE",
            "Price": price,
            "Shares": position,
            "Transaction_Fee": fee,
            "Cash": cash,
            "Portfolio_Value": cash
        })
        portfolio_history[-1] = cash
        position = 0
        
    test_data['Portfolio_Value'] = portfolio_history
    
    # 4. Calculate Benchmark (Buy and Hold)
    # Buy on first day of test set, sell on last day
    first_price = test_data.iloc[0]['Close']
    last_price = test_data.iloc[-1]['Close']
    
    bnh_shares = int((initial_cash * (1.0 - commission)) // first_price)
    bnh_cash = initial_cash - (bnh_shares * first_price) - (bnh_shares * first_price * commission)
    
    test_data['Benchmark_Value'] = bnh_cash + (bnh_shares * test_data['Close'])
    
    # 5. Compute Metrics
    final_ai_val = test_data.iloc[-1]['Portfolio_Value']
    final_bnh_val = test_data.iloc[-1]['Benchmark_Value']
    
    ai_total_return = (final_ai_val - initial_cash) / initial_cash
    bnh_total_return = (final_bnh_val - initial_cash) / initial_cash
    
    # Calculate daily returns for Sharpe Ratio
    test_data['Daily_Return'] = test_data['Portfolio_Value'].pct_change()
    mean_daily = test_data['Daily_Return'].mean()
    std_daily = test_data['Daily_Return'].std()
    
    # Annualized Sharpe (assuming risk-free rate = 6% annualized, which is ~0.00023 daily)
    daily_rf = 0.06 / 252
    if std_daily > 0 and not np.isnan(std_daily):
        sharpe = ((mean_daily - daily_rf) / std_daily) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Calculate Max Drawdown
    # Roll Max
    test_data['Roll_Max'] = test_data['Portfolio_Value'].cummax()
    test_data['Drawdown'] = (test_data['Portfolio_Value'] - test_data['Roll_Max']) / test_data['Roll_Max']
    max_drawdown = test_data['Drawdown'].min()
    
    # Win Rate
    win_rate = (profitable_trades / total_completed_trades) if total_completed_trades > 0 else 0.0
    
    metrics = {
        "Ticker": ticker_prefix,
        "Initial_Capital": initial_cash,
        "AI_Final_Value": final_ai_val,
        "Benchmark_Final_Value": final_bnh_val,
        "AI_Total_Return": ai_total_return,
        "Benchmark_Total_Return": bnh_total_return,
        "Annualized_Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_drawdown,
        "Total_Trades": len(trade_log),
        "Completed_Trades": total_completed_trades,
        "Win_Rate": win_rate
    }
    
    # Print Metrics Summary
    print("\n" + "=" * 60)
    print(f"            BACKTESTING REPORT: {ticker_prefix}")
    print("=" * 60)
    print(f"Test Period Rows:        {len(test_data)} trading days")
    print(f"Initial Capital:         {initial_cash:,.2f} INR")
    print(f"Final AI Portfolio Val:  {final_ai_val:,.2f} INR")
    print(f"Final B&H Benchmark Val: {final_bnh_val:,.2f} INR")
    print("-" * 60)
    print(f"AI Strategy Return:      {ai_total_return:.2%}")
    print(f"Buy & Hold Return:       {bnh_total_return:.2%}")
    print(f"Outperformance:          {ai_total_return - bnh_total_return:.2%}")
    print("-" * 60)
    print(f"Sharpe Ratio (Ann.):     {sharpe:.4f}")
    print(f"Max Drawdown (Risk):     {max_drawdown:.2%}")
    print(f"Total Trades Executed:   {len(trade_log)}")
    print(f"Completed Trades:        {total_completed_trades}")
    print(f"Strategy Win Rate:       {win_rate:.2%}")
    print("=" * 60 + "\n")
    
    # 6. Save files
    os.makedirs("data", exist_ok=True)
    trades_df = pd.DataFrame(trade_log)
    trades_file = os.path.join("data", f"{ticker_prefix}_backtest_trades.csv")
    trades_df.to_csv(trades_file, index=False)
    logging.info(f"[✓] Backtest trades logged to {trades_file}")
    
    # Save metrics key-value to CSV
    metrics_file = os.path.join("data", f"{ticker_prefix}_backtest_metrics.csv")
    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(metrics_file, index=False)
    
    # 7. Plot Equity Curves
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    dates_plot = pd.to_datetime(test_data['Date'])
    
    plt.plot(dates_plot, test_data['Portfolio_Value'], label='AI Strategy Portfolio', color='purple', linewidth=2)
    plt.plot(dates_plot, test_data['Benchmark_Value'], label='Buy & Hold Benchmark', color='orange', linestyle='--', linewidth=1.5)
    
    # Highlight trade entry/exits on the plot
    if trade_log:
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        buys = trades_df[trades_df['Action'] == 'BUY']
        sells = trades_df[trades_df['Action'].isin(['SELL', 'SELL_CLOSE'])]
        
        # Find portfolio values at trade dates
        buy_portfolio_vals = []
        for d in buys['Date']:
            val = test_data.loc[pd.to_datetime(test_data['Date']) == d, 'Portfolio_Value'].values
            buy_portfolio_vals.append(val[0] if len(val) > 0 else initial_cash)
            
        sell_portfolio_vals = []
        for d in sells['Date']:
            val = test_data.loc[pd.to_datetime(test_data['Date']) == d, 'Portfolio_Value'].values
            sell_portfolio_vals.append(val[0] if len(val) > 0 else initial_cash)
            
        # Draw markers on plot
        plt.scatter(buys['Date'], buy_portfolio_vals, marker='^', color='green', s=100, label='AI BUY Signal', zorder=5)
        plt.scatter(sells['Date'], sell_portfolio_vals, marker='v', color='red', s=100, label='AI SELL Signal', zorder=5)
        
    plt.title(f'AI Trading Backtest vs Benchmark - {ticker_prefix}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Equity (INR)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_file = os.path.join(plots_dir, f"{ticker_prefix}_backtest.png")
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"[✓] Equity curve chart saved to {plot_file}")
    
    return metrics

if __name__ == "__main__":
    # Test locally
    test_ticker = "RELIANCE"
    print(f"--- Running Backtest simulation for {test_ticker} ---")
    run_backtest(test_ticker)
