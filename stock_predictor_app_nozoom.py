import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("📈 Machine Learning Stock Price Predictor")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL)', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2019-01-01'))  # Longer range for 200-day SMA
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))
future_days = st.sidebar.slider('Number of Future Prediction Days', 1, 10, 5)

# Load data
def load_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

data = load_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data found. Please check the stock ticker or date range.")
    st.stop()

# Feature engineering with 100 & 200-day SMAs
def add_features(data):
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Return'] = data['Close'].pct_change()
    return data.dropna()

data = add_features(data)

# Prepare ML data
def prepare_data(data):
    X = data[['SMA_100', 'SMA_200', 'Return']]
    y = data['Close'].shift(-1)
    return train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(data)

# Train model
def train_and_predict(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test.values.flatten()) ** 2)
    return model, predictions, y_test, mse

model, predictions, y_test, mse = train_and_predict(X_train, X_test, y_train, y_test)

# Visualization
def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values.flatten(), label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# ✅ Recursive prediction with SMA freezing and constrained change
def recursive_predict(data, model, n_days=5):
    future_predictions = []
    last_data = data.copy()

    # Freeze indicators from the last real row
    sma_100_val = float(last_data['SMA_100'].iloc[-1])
    sma_200_val = float(last_data['SMA_200'].iloc[-1])
    prev_close_val = float(last_data['Close'].iloc[-1])
    last_return = last_data['Return'].iloc[-1]

    for i in range(n_days):
        # Use previous predicted return or fallback
        stock_return = (future_predictions[-1] - prev_close_val) / prev_close_val if i > 0 else last_return
        features_array = np.array([[sma_100_val, sma_200_val, stock_return]])

        raw_pred = model.predict(features_array)[0]

        # Constrain movement ±5%
        min_price = prev_close_val * 0.95
        max_price = prev_close_val * 1.05
        next_close = np.clip(raw_pred, min_price, max_price)

        future_predictions.append(next_close)
        prev_close_val = next_close  # update for next loop

    return future_predictions

# Display model performance
st.subheader(f"Model Performance for {ticker}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

# Show prediction chart
st.subheader("Price Prediction Visualization")
visualize_results(y_test, predictions)

# Show recent data
st.subheader("Recent Data Sample")
st.dataframe(data.tail(10))

# Future prediction
historical_dates = data.index
future_preds = recursive_predict(data, model, n_days=future_days)
future_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=future_days)

# Show predicted next close
st.subheader(f"🔮 Predicted Closing Price for {future_dates[0].date()}")
st.metric(label="Predicted Close", value=f"${float(future_preds[0]):,.2f}")

# --- Matplotlib Chart ---
st.subheader("📈 Historical and Future Predicted Prices")

plot_data = data[['Close', 'SMA_100', 'SMA_200']].dropna()
plot_dates = plot_data.index

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(plot_dates, plot_data['Close'], label='Historical Close', color='blue')
ax.plot(plot_dates, plot_data['SMA_100'], label='100-Day SMA', color='red', linestyle='--')
ax.plot(plot_dates, plot_data['SMA_200'], label='200-Day SMA', color='purple', linestyle='--')
ax.plot(future_dates, future_preds, label='Predicted Future Prices', color='green', linestyle='dashed', marker='o')

# Shaded area between last close and predictions
last_close = float(plot_data['Close'].iloc[-1])
baseline = np.full_like(future_preds, last_close)

ax.fill_between(
    future_dates,
    future_preds,
    baseline,
    color='green',
    alpha=0.1
)

ax.set_title(f"{ticker} Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Download forecast CSV
st.subheader("📥 Download Forecast Data")
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds})
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast as CSV", data=csv, file_name=f"{ticker}_forecast.csv", mime='text/csv')
