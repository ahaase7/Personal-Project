This Streamlit-based web app uses machine learning to predict future stock prices. It allows users to input a ticker symbol and date range, visualize historical stock data along with 100-day and 200-day moving averages, and view future closing price predictions using a trained Random Forest model.

🔧 Features
📅 User-defined date range and ticker input
📊 Technical indicators: 100-day & 200-day Simple Moving Averages (SMAs)
🤖 Machine Learning model (Random Forest Regressor) trained on historical features
🔁 Recursive prediction for short-term forecasts (up to 10 days)
📉 Stability constraints to avoid unrealistic market crashes
📷 Matplotlib visualizations for historical and future prices
📥 Downloadable CSV of forecasted values
🌐 Built with Streamlit for interactive UI


🧠 How It Works
Data Loading: Stock data is retrieved from Yahoo Finance via yfinance.
Feature Engineering: Calculates 100-day and 200-day SMAs and daily returns.
Model Training: A Random Forest model is trained to predict the next day's closing price using engineered features.
Prediction: Future prices are recursively forecasted using the model, with constrained daily change to ±5%.
Visualization: Uses matplotlib to plot historical data, SMAs, and future predictions.

Install Dependencies
streamlit
yfinance
pandas, numpy
matplotlib
scikit-learn

File Structure
├── stock_predictor_app_nozoom.py     # Main Streamlit app
├── README.md                         # Project description
├── requirements.txt                  # Python dependencies
