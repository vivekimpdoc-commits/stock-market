import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Dimensions,
} from 'react-native';

const { width } = Dimensions.get('window');

/**
 * A highly polished React Native Component that integrates with the FastAPI Stock Market AI server.
 * Fetches real-time price forecasting, news sentiment, and AI recommendations.
 * 
 * To use this in your React Native app:
 * 1. Install dependencies: npm install react-native-svg (if using advanced charts)
 * 2. Import this file and render it in your App.js or navigation screens.
 * 3. Make sure to point the API_BASE_URL to your server's IP address.
 */
export default function StockPredictorApp() {
  // CONFIGURATION: Set your FastAPI server IP here.
  // 10.0.2.2 is the default IP to access localhost on Android Emulator.
  // For iOS Simulator, use 'http://127.0.0.1:8000'.
  // For physical devices, use your computer's local IP (e.g., 'http://192.168.1.100:8000').
  const API_BASE_URL = 'http://10.0.2.2:8000';

  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [liveQuote, setLiveQuote] = useState(null);
  const [sentiment, setSentiment] = useState(null);

  // Fetch prediction, live quote, and sentiment simultaneously
  const handleSearch = async () => {
    if (!ticker.trim()) return;
    
    setLoading(true);
    setError(null);
    setPrediction(null);
    setLiveQuote(null);
    setSentiment(null);

    const formattedTicker = ticker.trim().toUpperCase();

    try {
      // 1. Fetch Price Trend Prediction
      const predResponse = await fetch(`${API_BASE_URL}/predict/${formattedTicker}`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });

      if (!predResponse.ok) {
        const errorData = await predResponse.json();
        throw new Error(errorData.detail || `Prediction error (${predResponse.status})`);
      }
      const predData = await predResponse.json();
      setPrediction(predData);

      // 2. Fetch Live Quote (Non-blocking fallback)
      try {
        const liveResponse = await fetch(`${API_BASE_URL}/api/live/quote/${formattedTicker}`, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        });
        if (liveResponse.ok) {
          const liveData = await liveResponse.json();
          setLiveQuote(liveData);
        }
      } catch (liveErr) {
        console.warn('Failed to fetch live quote info:', liveErr);
      }

      // 3. Fetch News Sentiment (Non-blocking fallback)
      try {
        const sentResponse = await fetch(`${API_BASE_URL}/sentiment/${formattedTicker}`, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        });
        if (sentResponse.ok) {
          const sentData = await sentResponse.json();
          setSentiment(sentData);
        }
      } catch (sentimentErr) {
        console.warn('Failed to fetch sentiment info:', sentimentErr);
      }

    } catch (err) {
      setError(err.message || 'Connection failed. Ensure the FastAPI server is running.');
    } finally {
      setLoading(false);
    }
  };

  const isBuy = prediction?.prediction_code === 1;
  const accentColor = isBuy ? '#10B981' : '#EF4444'; // Emerald / Red
  const glowColor = isBuy ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0B0F19" />
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>StockMarket AI</Text>
          <Text style={styles.headerSubtitle}>Real-time Predictive Analytics</Text>
        </View>

        {/* Search Bar Card */}
        <View style={styles.searchCard}>
          <TextInput
            style={styles.input}
            placeholder="Enter Stock Ticker (e.g. RELIANCE)"
            placeholderTextColor="#6B7280"
            value={ticker}
            onChangeText={setTicker}
            autoCapitalize="characters"
            autoCorrect={false}
            onSubmitEditing={handleSearch}
          />
          <TouchableOpacity 
            style={[styles.searchButton, !ticker.trim() && styles.searchButtonDisabled]} 
            onPress={handleSearch}
            disabled={!ticker.trim() || loading}
          >
            <Text style={styles.searchButtonText}>Analyze</Text>
          </TouchableOpacity>
        </View>

        {/* Loading Indicator */}
        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#8B5CF6" />
            <Text style={styles.loadingText}>Running AI Model calculations...</Text>
          </View>
        )}

        {/* Error Alert */}
        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Prediction Results */}
        {prediction && !loading && (
          <View style={[styles.resultCard, { borderColor: accentColor + '3F', backgroundColor: 'rgba(17, 25, 40, 0.85)' }]}>
            
            {/* Ticker Name */}
            <View style={styles.tickerHeader}>
              <Text style={styles.tickerText}>{prediction.ticker}</Text>
              <Text style={styles.dateText}>Predicted on {prediction.prediction_date}</Text>
            </View>

            {/* Last Close Price */}
            <Text style={styles.priceText}>
              Last Close Price: <Text style={styles.boldText}>₹{prediction.last_close_price?.toFixed(2)}</Text>
            </Text>

            {/* Live Price LTP (if available) */}
            {liveQuote && (
              <View style={styles.liveQuoteContainer}>
                <View style={styles.liveDotRow}>
                  <View style={styles.liveDot} />
                  <Text style={styles.liveLabel}>LIVE PRICE (LTP)</Text>
                </View>
                <Text style={styles.livePriceText}>
                  ₹{liveQuote.lastPrice?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                  <Text style={[styles.liveChangeText, { color: liveQuote.change >= 0 ? '#10B981' : '#EF4444' }]}>
                    {' '}{liveQuote.change >= 0 ? '+' : ''}{liveQuote.change?.toFixed(2)} ({liveQuote.change >= 0 ? '+' : ''}{liveQuote.pChange?.toFixed(2)}%)
                  </Text>
                </Text>
                <Text style={styles.liveDetailsText}>
                  High: ₹{liveQuote.dayHigh?.toLocaleString('en-IN', { minimumFractionDigits: 2 })} | Low: ₹{liveQuote.dayLow?.toLocaleString('en-IN', { minimumFractionDigits: 2 })} | Vol: {liveQuote.volume?.toLocaleString('en-IN')}
                </Text>
                <Text style={styles.liveSourceText}>
                  Source: {liveQuote.source}
                </Text>
              </View>
            )}

            <View style={styles.divider} />

            {/* Recommendation Display */}
            <Text style={styles.sectionLabel}>AI FORECAST ACTION</Text>
            <Text style={[styles.recommendationText, { color: accentColor }]}>
              {prediction.recommendation}
            </Text>
            
            <Text style={styles.detailsText}>{prediction.details}</Text>

            <View style={styles.divider} />

            {/* Confidence progress */}
            <View style={styles.confidenceContainer}>
              <View style={styles.confidenceHeader}>
                <Text style={styles.confidenceLabel}>Model Confidence</Text>
                <Text style={[styles.confidenceValue, { color: accentColor }]}>
                  {(prediction.confidence_score * 100).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.progressBarBackground}>
                <View 
                  style={[
                    styles.progressBarFill, 
                    { 
                      width: `${prediction.confidence_score * 100}%`, 
                      backgroundColor: accentColor 
                    }
                  ]} 
                />
              </View>
            </View>
          </View>
        )}

        {/* Sentiment Analysis Panel */}
        {sentiment && !loading && (
          <View style={styles.sentimentCard}>
            <Text style={styles.sectionLabel}>LIVE HEADLINE SENTIMENT</Text>
            
            <View style={styles.sentimentOverview}>
              <View>
                <Text style={styles.sentimentLabel}>Overall Sentiment</Text>
                <Text style={[
                  styles.sentimentSummaryText,
                  { color: sentiment.mean_compound_score >= 0.05 ? '#10B981' : sentiment.mean_compound_score <= -0.05 ? '#EF4444' : '#F59E0B' }
                ]}>
                  {sentiment.sentiment_summary}
                </Text>
              </View>
              <View style={styles.scoreBadge}>
                <Text style={styles.scoreBadgeText}>
                  Score: {sentiment.mean_compound_score?.toFixed(2)}
                </Text>
              </View>
            </View>

            {sentiment.headlines && sentiment.headlines.length > 0 ? (
              <View style={styles.headlinesList}>
                <Text style={styles.headlinesHeader}>Recent Articles ({sentiment.news_found}):</Text>
                {sentiment.headlines.slice(0, 3).map((headline, idx) => (
                  <View key={idx} style={styles.headlineItem}>
                    <Text style={styles.headlineBullet}>•</Text>
                    <Text style={styles.headlineText} numberOfLines={2}>
                      {headline}
                    </Text>
                  </View>
                ))}
              </View>
            ) : (
              <Text style={styles.noHeadlinesText}>No recent news articles found for this ticker.</Text>
            )}
          </View>
        )}

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0F19', // Deep dark backdrop
  },
  scrollContent: {
    padding: 20,
    alignItems: 'stretch',
  },
  header: {
    marginTop: 20,
    marginBottom: 25,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: '#FFFFFF',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#9CA3AF',
    marginTop: 4,
    fontWeight: '400',
  },
  searchCard: {
    flexDirection: 'row',
    backgroundColor: '#1F2937',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    alignItems: 'center',
    marginBottom: 20,
  },
  input: {
    flex: 1,
    height: 48,
    color: '#FFFFFF',
    fontSize: 15,
    paddingHorizontal: 8,
  },
  searchButton: {
    backgroundColor: '#3B82F6',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  searchButtonDisabled: {
    backgroundColor: '#1F2937',
    opacity: 0.5,
  },
  searchButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  loadingContainer: {
    marginVertical: 40,
    alignItems: 'center',
  },
  loadingText: {
    color: '#9CA3AF',
    fontSize: 14,
    marginTop: 12,
  },
  errorCard: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
    padding: 16,
    marginBottom: 20,
  },
  errorText: {
    color: '#FCA5A5',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  resultCard: {
    borderRadius: 16,
    borderWidth: 1,
    padding: 24,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 15,
    elevation: 8,
  },
  tickerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 6,
  },
  tickerText: {
    fontSize: 24,
    fontWeight: '800',
    color: '#FFFFFF',
  },
  dateText: {
    fontSize: 11,
    color: '#9CA3AF',
  },
  priceText: {
    fontSize: 15,
    color: '#D1D5DB',
  },
  boldText: {
    fontWeight: '700',
    color: '#FFFFFF',
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginVertical: 20,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: '#9CA3AF',
    letterSpacing: 1.5,
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  recommendationText: {
    fontSize: 40,
    fontWeight: '900',
    letterSpacing: 2,
    marginBottom: 10,
  },
  detailsText: {
    fontSize: 14,
    color: '#E5E7EB',
    lineHeight: 22,
  },
  confidenceContainer: {
    marginTop: 5,
  },
  confidenceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  confidenceLabel: {
    fontSize: 13,
    color: '#9CA3AF',
  },
  confidenceValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  progressBarBackground: {
    height: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 5,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 5,
  },
  sentimentCard: {
    backgroundColor: '#111928',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    padding: 24,
    marginBottom: 30,
  },
  sentimentOverview: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  sentimentSummaryText: {
    fontSize: 20,
    fontWeight: '700',
    marginTop: 2,
  },
  scoreBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 8,
    paddingVertical: 6,
    paddingHorizontal: 12,
  },
  scoreBadgeText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '500',
  },
  headlinesList: {
    marginTop: 10,
  },
  headlinesHeader: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9CA3AF',
    marginBottom: 10,
  },
  headlineItem: {
    flexDirection: 'row',
    marginBottom: 8,
    paddingRight: 10,
  },
  headlineBullet: {
    color: '#8B5CF6',
    fontSize: 14,
    fontWeight: 'bold',
    marginRight: 8,
  },
  headlineText: {
    color: '#D1D5DB',
    fontSize: 13,
    lineHeight: 18,
    flex: 1,
  },
  noHeadlinesText: {
    color: '#9CA3AF',
    fontSize: 13,
    fontStyle: 'italic',
    marginTop: 8,
  },
  liveQuoteContainer: {
    marginTop: 12,
    padding: 14,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  liveDotRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#10B981',
    marginRight: 6,
  },
  liveLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: '#9CA3AF',
    letterSpacing: 1,
  },
  livePriceText: {
    fontSize: 18,
    fontWeight: '800',
    color: '#FFFFFF',
  },
  liveChangeText: {
    fontSize: 13,
    fontWeight: '600',
  },
  liveDetailsText: {
    fontSize: 11,
    color: '#9CA3AF',
    marginTop: 4,
  },
  liveSourceText: {
    fontSize: 9,
    color: '#F59E0B',
    fontWeight: '600',
    textTransform: 'uppercase',
    marginTop: 2,
  },
});
