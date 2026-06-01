import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_finbert_sentiment(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Loads financial news headlines from a CSV, runs the FinBERT sentiment analysis model,
    compares results against VADER sentiment, and saves the output.
    
    If PyTorch or Transformers are missing, falls back to a simulated placeholder
    so the pipeline does not break out-of-the-box, providing instructions for installation.
    """
    if not os.path.exists(input_csv):
        logging.error(f"Input news CSV file '{input_csv}' not found. Fetch news first.")
        return pd.DataFrame()
        
    logging.info(f"Reading news headlines from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if df.empty or 'title' not in df.columns:
        logging.error("Input CSV is empty or is missing the 'title' column.")
        return pd.DataFrame()
        
    # Attempt imports of Hugging Face Transformers and PyTorch
    try:
        logging.info("Importing PyTorch and Transformers (Hugging Face)...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        
        logging.info("Loading pre-trained FinBERT model ('ProsusAI/finbert')...")
        logging.info("Note: This might take a minute on the first run to download the ~400MB model.")
        
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Initialize classifier pipeline
        # Use GPU (device=0) if CUDA is available, else CPU (device=-1)
        device = 0 if torch.cuda.is_available() else -1
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        
        is_fallback = False
    except ImportError:
        logging.warning("\n" + "!" * 80)
        logging.warning("Hugging Face 'transformers' or 'torch' is not installed.")
        logging.warning("Please install them using: pip install torch transformers")
        logging.warning("Running in VADER/Rule-based FinBERT simulation fallback mode...")
        logging.warning("!" * 80 + "\n")
        is_fallback = True
        
    finbert_labels = []
    finbert_scores = []
    
    # Process headlines
    logging.info(f"Analyzing {len(df)} headlines with {'FinBERT' if not is_fallback else 'Simulation Fallback'}...")
    
    if not is_fallback:
        try:
            # Batch process for speed
            titles = df['title'].tolist()
            # Truncate strings to prevent token length errors in long paragraphs
            titles = [str(t)[:512] for t in titles]
            
            results = nlp(titles)
            for res in results:
                # FinBERT outputs labels as: positive, negative, neutral (lowercase)
                label = res['label'].capitalize()
                score = res['score']
                finbert_labels.append(label)
                finbert_scores.append(score)
        except Exception as e:
            logging.error(f"Error executing FinBERT pipeline: {e}")
            logging.warning("Reverting to simulation fallback...")
            is_fallback = True
            
    if is_fallback:
        # FinBERT simulation fallback using simple financial keywords and VADER alignment
        for index, row in df.iterrows():
            title_lower = str(row['title']).lower()
            
            # Simple rule-based logic for financial context
            if any(w in title_lower for w in ['gain', 'profit up', 'surge', 'rise', 'record high', 'growth', 'bull', 'jump']):
                label = "Positive"
                score = 0.85
            elif any(w in title_lower for w in ['slip', 'loss', 'drop', 'fall', 'slump', 'plunge', 'decline', 'bear', 'deficit']):
                label = "Negative"
                score = 0.88
            else:
                # Align with VADER score if present
                vader_label = row.get('sentiment_label', 'Neutral')
                label = vader_label
                score = abs(row.get('sentiment_compound', 0.5))
                
            finbert_labels.append(label)
            finbert_scores.append(score)
            
    # Add columns to DataFrame
    df['finbert_label'] = finbert_labels
    df['finbert_score'] = finbert_scores
    
    # Save output
    if not output_csv:
        dir_name = os.path.dirname(input_csv)
        base_name = os.path.basename(input_csv)
        output_csv = os.path.join(dir_name, base_name.replace("market_sentiment", "finbert_sentiment"))
        if output_csv == input_csv:
            output_csv = os.path.join(dir_name, "finbert_" + base_name)
            
    df.to_csv(output_csv, index=False)
    logging.info(f"[✓] Sentiment data saved successfully to {output_csv}")
    
    # Print comparison summary (VADER vs FinBERT)
    print("\n" + "=" * 60)
    print("           SENTIMENT MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total Headlines Analyzed: {len(df)}")
    print("-" * 60)
    
    if 'sentiment_label' in df.columns:
        print("VADER Sentiment Distribution:")
        print(df['sentiment_label'].value_counts().to_string())
        print("-" * 60)
        
    print("FinBERT (or Fallback) Sentiment Distribution:")
    print(df['finbert_label'].value_counts().to_string())
    print("=" * 60 + "\n")
    
    # Sample comparison rows
    sample_cols = ['title', 'sentiment_label', 'finbert_label'] if 'sentiment_label' in df.columns else ['title', 'finbert_label']
    print("Sample Comparison Rows:")
    print(df[sample_cols].head(3).to_string())
    print("=" * 60 + "\n")
    
    return df

if __name__ == "__main__":
    # Test script locally
    test_file = "data/market_sentiment.csv"
    if os.path.exists(test_file):
        print(f"--- Running FinBERT Sentiment Analysis for {test_file} ---")
        run_finbert_sentiment(test_file)
    else:
        print("[!] No default market sentiment file found. Run news fetching first.")
