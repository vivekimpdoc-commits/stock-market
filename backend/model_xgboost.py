import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_xgboost_model(ticker_prefix: str) -> bool:
    """
    Loads preprocessed classification datasets, trains an XGBoost classifier
    (with RandomForest fallback if xgboost is unavailable), prints performance
    metrics, and plots feature importance.
    
    Parameters:
    - ticker_prefix: The prefix name of the stock file (e.g., RELIANCE)
    
    Returns:
    - Success status boolean
    """
    data_dir = "data/preprocessed"
    models_dir = "models"
    plots_dir = "plots"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    x_train_path = os.path.join(data_dir, f"{ticker_prefix}_X_train.csv")
    x_test_path = os.path.join(data_dir, f"{ticker_prefix}_X_test.csv")
    y_train_path = os.path.join(data_dir, f"{ticker_prefix}_y_train_clf.csv")
    y_test_path = os.path.join(data_dir, f"{ticker_prefix}_y_test_clf.csv")
    
    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            logging.error(f"Required preprocessing file '{path}' is missing. Run preprocessing first.")
            return False
            
    logging.info(f"Loading datasets for {ticker_prefix}...")
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train_df = pd.read_csv(y_train_path)
    y_test_df = pd.read_csv(y_test_path)
    
    feature_names = [col for col in X_train_df.columns if col != 'Date']
    
    # Drop Date column, get array values
    X_train = X_train_df[feature_names].values
    X_test = X_test_df[feature_names].values
    
    y_train = y_train_df['Target_Direction'].values
    y_test = y_test_df['Target_Direction'].values
    
    # Try importing XGBoost
    try:
        import xgboost as xgb
        logging.info("Imported XGBoost. Initializing XGBClassifier...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )
        is_xgb = True
    except ImportError:
        logging.warning("XGBoost is not installed. Falling back to RandomForestClassifier...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        is_xgb = False
        
    logging.info("Training classification model...")
    model.fit(X_train, y_train)
    
    # Save the trained model
    if is_xgb:
        model_path = os.path.join(models_dir, f"{ticker_prefix}_xgboost_model.json")
        model.save_model(model_path)
    else:
        model_path = os.path.join(models_dir, f"{ticker_prefix}_randomforest_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    logging.info(f"[✓] Model saved successfully to {model_path}")
    
    # Evaluate
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print("\n" + "=" * 50)
    print(f"       CLASSIFICATION EVALUATION: {ticker_prefix}")
    print("=" * 50)
    print(f"Accuracy Score:  {accuracy:.4%}")
    print(f"Precision Score: {precision:.4%}")
    print(f"Recall Score:    {recall:.4%}")
    print(f"F1-Score:        {f1:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("=" * 50 + "\n")
    
    # Plot feature importance
    logging.info("Plotting Feature Importance...")
    if is_xgb:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_
        
    # Sort importances
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Take top 10 features for cleaner plot
    top_n = min(10, len(sorted_features))
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), sorted_importances[:top_n][::-1], align='center', color='skyblue')
    plt.yticks(range(top_n), sorted_features[:top_n][::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Features for {ticker_prefix} Price Direction Prediction')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, f"{ticker_prefix}_feature_importance.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"[✓] Feature importance plot saved to {plot_path}")
    
    return True

if __name__ == "__main__":
    # Test locally
    test_ticker = "RELIANCE"
    print(f"--- Running Classification test training for {test_ticker} ---")
    train_xgboost_model(test_ticker)
