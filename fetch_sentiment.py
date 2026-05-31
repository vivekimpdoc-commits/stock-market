import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Standard RSS feeds for financial news in India
RSS_FEEDS = {
    "Moneycontrol_Latest": "https://www.moneycontrol.com/rss/latestnews.xml",
    "Moneycontrol_Business": "https://www.moneycontrol.com/rss/business.xml",
    "EconomicTimes_Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "EconomicTimes_TopStories": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    "LiveMint_Markets": "https://www.livemint.com/rss/markets",
    "LiveMint_News": "https://www.livemint.com/rss/news"
}

def parse_rss_feed(feed_name: str, url: str) -> list:
    """
    Fetches and parses an RSS feed, returning a list of news dictionaries.
    """
    logging.info(f"Fetching RSS feed: {feed_name}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    news_items = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logging.error(f"Failed to fetch {feed_name}: HTTP {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all("item")
        
        for item in items:
            title = item.find("title").text.strip() if item.find("title") else ""
            link = item.find("link").text.strip() if item.find("link") else ""
            pub_date_str = item.find("pubDate").text.strip() if item.find("pubDate") else ""
            desc = item.find("description").text.strip() if item.find("description") else ""
            
            # Remove HTML tags from description if any
            if desc:
                desc = BeautifulSoup(desc, "html.parser").get_text().strip()
                
            # Try to parse date
            pub_date = None
            if pub_date_str:
                for fmt in ('%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S GMT', '%d %b %Y %H:%M:%S %z', '%Y-%m-%d %H:%M:%S'):
                    try:
                        # Clean up timezone strings that python might struggle with
                        cleaned_date = pub_date_str.replace("IST", "+0530")
                        pub_date = datetime.strptime(cleaned_date, fmt)
                        break
                    except ValueError:
                        continue
            
            # Format date for consistency
            formatted_date = pub_date.strftime('%Y-%m-%d %H:%M:%S') if pub_date else pub_date_str
            
            news_items.append({
                "source": feed_name,
                "title": title,
                "description": desc,
                "published_at": formatted_date,
                "url": link
            })
            
        logging.info(f"Successfully parsed {len(news_items)} items from {feed_name}.")
    except Exception as e:
        logging.error(f"Error parsing RSS feed {feed_name}: {e}")
        
    return news_items

def analyze_sentiment(news_items: list) -> pd.DataFrame:
    """
    Computes VADER sentiment scores for a list of news items.
    Adds scores: Positive, Negative, Neutral, and Compound.
    """
    if not news_items:
        return pd.DataFrame()
        
    logging.info("Analyzing sentiments using VADER...")
    analyzer = SentimentIntensityAnalyzer()
    
    scored_items = []
    for item in news_items:
        # We analyze the headline (title) since it holds the strongest immediate message.
        # Description can be added if available.
        text_to_analyze = item["title"]
        if item["description"] and len(item["description"]) > 10:
            text_to_analyze += ". " + item["description"]
            
        scores = analyzer.polarity_scores(text_to_analyze)
        
        # Determine overall sentiment class based on compound score
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment_label = "Positive"
        elif compound <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        scored_item = {
            **item,
            "sentiment_positive": scores["pos"],
            "sentiment_negative": scores["neg"],
            "sentiment_neutral": scores["neu"],
            "sentiment_compound": scores["compound"],
            "sentiment_label": sentiment_label
        }
        scored_items.append(scored_item)
        
    return pd.DataFrame(scored_items)

def fetch_and_analyze_sentiment(keyword_filter: str = None, save_dir: str = "data") -> pd.DataFrame:
    """
    Fetches all configured RSS feeds, computes sentiment, filters by keyword if provided,
    and saves to a CSV.
    
    Parameters:
    - keyword_filter: Filter news where title/description contains this keyword (case-insensitive).
    - save_dir: Directory where the CSV will be saved.
    
    Returns:
    - DataFrame of scored news headlines.
    """
    all_news = []
    for name, url in RSS_FEEDS.items():
        all_news.extend(parse_rss_feed(name, url))
        
    if not all_news:
        logging.warning("No news fetched from any source.")
        return pd.DataFrame()
        
    df = analyze_sentiment(all_news)
    
    # Filter by keyword if requested
    if keyword_filter and not df.empty:
        logging.info(f"Filtering news containing keyword: '{keyword_filter}'...")
        # Check both title and description
        keyword_lower = keyword_filter.lower()
        mask = df['title'].str.lower().str.contains(keyword_lower) | df['description'].str.lower().str.contains(keyword_lower)
        df = df[mask]
        logging.info(f"Found {len(df)} matching news articles.")
        
    if df.empty:
        logging.warning("No headlines matched filter criteria or no articles found.")
        return df
        
    # Ensure target directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Build path and save
    suffix = f"_{keyword_filter.replace(' ', '_')}" if keyword_filter else ""
    file_path = os.path.join(save_dir, f"market_sentiment{suffix}.csv")
    df.to_csv(file_path, index=False)
    logging.info(f"Saved news sentiment data to {file_path}")
    
    return df

if __name__ == "__main__":
    # Test script locally
    print("--- Fetching All Latest Financial News & Analyzing Sentiment ---")
    df_all = fetch_and_analyze_sentiment()
    if not df_all.empty:
        print(df_all[['source', 'title', 'sentiment_compound', 'sentiment_label']].head(5))
        
    print("\n--- Fetching News Filtered for 'Reliance' ---")
    df_filtered = fetch_and_analyze_sentiment(keyword_filter="Reliance")
    if not df_filtered.empty:
        print(df_filtered[['source', 'title', 'sentiment_compound', 'sentiment_label']].head(5))
