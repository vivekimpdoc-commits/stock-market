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
            
        try:
            soup = BeautifulSoup(response.content, features="xml")
        except Exception:
            # Fallback to standard html.parser if xml features (like lxml) aren't available
            soup = BeautifulSoup(response.content, "html.parser")
            
        items = soup.find_all("item")
        
        for item in items:
            title = item.find("title").text.strip() if item.find("title") else ""
            link = item.find("link").text.strip() if item.find("link") else ""
            pub_date_tag = item.find("pubDate") or item.find("pubdate")
            pub_date_str = pub_date_tag.text.strip() if pub_date_tag else ""
            desc = item.find("description").text.strip() if item.find("description") else ""
            
            # Extract source tag if present (common in Google News RSS)
            source_tag = item.find("source")
            source_name = source_tag.text.strip() if source_tag else ""
            
            # Determine source label
            if source_name:
                item_source = f"{feed_name} ({source_name})"
            else:
                item_source = feed_name
            
            # Remove HTML tags from description if any
            if desc:
                desc = BeautifulSoup(desc, "html.parser").get_text().strip()
                
            # Try to parse date
            pub_date = None
            if pub_date_str:
                for fmt in (
                    '%a, %d %b %Y %H:%M:%S %z',
                    '%a, %d %b %Y %H:%M:%S GMT',
                    '%a, %d %b %Y %H:%M:%S',
                    '%d %b %Y %H:%M:%S %z',
                    '%d %b %Y %H:%M:%S GMT',
                    '%d %b %Y %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S%z',
                    '%Y-%m-%d'
                ):
                    try:
                        # Clean up timezone strings that python might struggle with
                        cleaned_date = pub_date_str.replace("IST", "+0530")
                        pub_date = datetime.strptime(cleaned_date, fmt)
                        if pub_date.tzinfo is not None:
                            pub_date = pub_date.replace(tzinfo=None)
                        break
                    except ValueError:
                        continue
            
            # Filter for last 7 days if it's Google News
            if "googlenews" in feed_name.lower() and pub_date:
                time_diff = datetime.now() - pub_date
                if time_diff.days > 7:
                    continue
            
            # Format date for consistency
            formatted_date = pub_date.strftime('%Y-%m-%d %H:%M:%S') if pub_date else pub_date_str
            
            news_items.append({
                "source": item_source,
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
    Fetches all configured RSS feeds, including Google News search feed, computes sentiment,
    filters by keyword if provided, sorts them chronologically, and saves to a CSV.
    
    Parameters:
    - keyword_filter: Filter news where title/description contains this keyword (case-insensitive).
    - save_dir: Directory where the CSV will be saved.
    
    Returns:
    - DataFrame of scored news headlines.
    """
    all_news = []
    
    # 1. Fetch standard news RSS feeds
    for name, url in RSS_FEEDS.items():
        all_news.extend(parse_rss_feed(name, url))
        
    # 2. Fetch Google News search feed
    google_query = keyword_filter if keyword_filter else "Indian stock market"
    google_url = f"https://news.google.com/rss/search?q={google_query}&hl=en-IN&gl=IN&ceid=IN:en"
    all_news.extend(parse_rss_feed("GoogleNews", google_url))
        
    if not all_news:
        logging.warning("No news fetched from any source.")
        return pd.DataFrame()
        
    df = analyze_sentiment(all_news)
    
    # 3. Filter by keyword if requested (bypassing Google News search results as they are already filtered)
    if keyword_filter and not df.empty:
        logging.info(f"Filtering news containing keyword: '{keyword_filter}'...")
        keyword_lower = keyword_filter.lower()
        is_google = df['source'].str.startswith('GoogleNews')
        matches_keyword = df['title'].str.lower().str.contains(keyword_lower) | df['description'].str.lower().str.contains(keyword_lower)
        df = df[is_google | matches_keyword]
        logging.info(f"Found {len(df)} matching news articles (including Google News).")
        
    if df.empty:
        logging.warning("No headlines matched filter criteria or no articles found.")
        return df
        
    # 4. Sort the DataFrame chronologically (newest first)
    df['temp_date'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.sort_values(by='temp_date', ascending=False, na_position='last')
    df = df.drop(columns=['temp_date'])
        
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
