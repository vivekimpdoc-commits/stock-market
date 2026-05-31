import 'dart:convert';
import 'package:flutter/material';
import 'package:http/http.dart' as http;

/// A Flutter widget that integrates with the FastAPI Stock Market AI server.
/// Fetches real-time price forecasting and recommendation for any ticker.
class StockPredictorWidget extends StatefulWidget {
  final String apiBaseUrl; // e.g., 'http://10.0.2.2:8000' (default Android emulator localhost)

  const StockPredictorWidget({
    Key? key,
    this.apiBaseUrl = 'http://10.0.2.2:8000', // Uses 10.0.2.2 for Android Emulator, use 127.0.0.1 for iOS
  }) : super(key: key);

  @override
  _StockPredictorWidgetState createState() => _StockPredictorWidgetState();
}

class _StockPredictorWidgetState extends State<StockPredictorWidget> {
  final TextEditingController _tickerController = TextEditingController();
  bool _isLoading = false;
  Map<String, dynamic>? _predictionData;
  String? _errorMessage;

  // Function to call FastAPI GET /predict/{ticker}
  Future<void> fetchPrediction(String ticker) async {
    if (ticker.isEmpty) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _predictionData = null;
    });

    final url = Uri.parse('${widget.apiBaseUrl}/predict/\$ticker');

    try {
      final response = await http.get(url).timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        setState(() {
          _predictionData = json.decode(response.body);
          _isLoading = false;
        });
      } else {
        final errorMsg = json.decode(response.body)['detail'] ?? 'Failed to fetch predictions';
        setState(() {
          _errorMessage = errorMsg;
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Network error: Could not connect to API server. \\nMake sure Uvicorn is running.';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // Styling Colors
    final isBuy = _predictionData?['prediction_code'] == 1;
    final accentColor = isBuy ? Colors.emerald : Colors.redAccent;
    final glowColor = isBuy ? Colors.emerald.withOpacity(0.1) : Colors.red.withOpacity(0.1);

    return Scaffold(
      backgroundColor: const Color(0xFF0B0F19), // Deep dark theme background
      appBar: AppBar(
        title: const Text('StockMarket AI Predictor', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF111928),
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Search Input Card
            Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: const Color(0xFF1F2937),
                borderRadius: BorderRadius.circular(15),
                border: Border.all(color: Colors.white.withOpacity(0.08)),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _tickerController,
                      style: const TextStyle(color: Colors.white),
                      decoration: const InputDecoration(
                        hintText: 'Enter Stock Ticker (e.g. RELIANCE)',
                        hintStyle: TextStyle(color: Colors.grey),
                        border: InputBorder.none,
                      ),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: () => fetchPrediction(_tickerController.text.trim()),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF3B82F6),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    child: const Text('Search'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 25),

            // Loading Indicator
            if (_isLoading)
              const Center(
                child: Padding(
                  padding: EdgeInsets.symmetric(vertical: 40),
                  child: CircularProgressIndicator(color: Colors.purpleAccent),
                ),
              ),

            // Error Message
            if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.red.withOpacity(0.3)),
                ),
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.redAccent, fontSize: 14),
                  textAlign: TextAlign.center,
                ),
              ),

            // Prediction Results Display
            if (_predictionData != null && !_isLoading)
              AnimatedContainer(
                duration: const Duration(milliseconds: 500),
                padding: const EdgeInsets.all(25),
                decoration: BoxDecoration(
                  color: const Color(0xFF111928).withOpacity(0.85),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: accentColor.withOpacity(0.3)),
                  boxShadow: [
                    BoxShadow(
                      color: glowColor,
                      blurRadius: 20,
                      spreadRadius: 2,
                    )
                  ],
                ),
                child: Column(
                  children: [
                    Text(
                      _predictionData!['ticker'],
                      style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 5),
                    Text(
                      'Last Close: ₹ \${_predictionData!['last_close_price'].toStringAsFixed(2)}',
                      style: const TextStyle(color: Colors.grey, fontSize: 16),
                    ),
                    const Divider(color: Colors.grey, height: 40, thickness: 0.5),
                    
                    const Text(
                      'AI FORECAST ACTION',
                      style: TextStyle(color: Colors.grey, fontSize: 12, letterSpacing: 1.5, fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      _predictionData!['recommendation'],
                      style: TextStyle(
                        color: accentColor,
                        fontSize: 42,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 2,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      _predictionData!['details'],
                      style: const TextStyle(color: Colors.white70, fontSize: 14),
                      textAlign: TextAlign.center,
                    ),
                    
                    const SizedBox(height: 30),
                    // Confidence Bar
                    Text(
                      'Model Confidence: \${(_predictionData!['confidence_score'] * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(color: Colors.grey, fontSize: 13, fontWeight: FontWeight.w500),
                    ),
                    const SizedBox(height: 10),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: LinearProgressIndicator(
                        value: _predictionData!['confidence_score'],
                        backgroundColor: Colors.white.withOpacity(0.05),
                        color: accentColor,
                        minHeight: 12,
                      ),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      'Prediction Date: \${_predictionData!['prediction_date']}',
                      style: const TextStyle(color: Colors.grey, fontSize: 11),
                    )
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
