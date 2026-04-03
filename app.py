import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

# ----------------------
# 🔧 MODEL CONFIG: STOCK → MODEL MAPPING
# ----------------------
# 👉 Edit this part only when you add/change stocks/models
MODEL_CONFIG = {
    "TATAMOTORS.NS": {
        "model_type": "ensemble",
        "model_path": "models/tatamotors_ensemble.pkl",
        "scaler_path": "models/TATAMOTORS.NS_scaler.pkl",
        "label": "Ensemble (RF + XGB + ...)"
    },
    "RELIANCE.NS": {
        "model_type": "ensemble",
        "model_path": "models/reliance_ensemble.pkl",
        "scaler_path": "models/RELIANCE.NS_scaler.pkl",
        "label": "Ensemble (RF + XGB + ...)"
    },

    # EXAMPLES – CHANGE PATHS TO YOUR ACTUAL MODEL FILES
    "HDFCBANK.NS": {
        "model_type": "xgboost",
        "model_path": "models/HDFCBANK.NS_xgb_model.pkl",
        "scaler_path": "models/HDFCBANK.NS_scaler.pkl",
        "label": "XGBoost"
    },
    "INFY.NS": {
        "model_type": "random_forest",
        "model_path": "models/INFY.NS_rf_model.pkl",
        "scaler_path": "models/INFY.NS_scaler.pkl",
        "label": "Random Forest"
    },
    "ICICIBANK.NS": {
        "model_type": "random_forest",
        "model_path": "models/ICICIBANK.NS_rf_model.pkl",
        "scaler_path": "models/ICICIBANK.NS_scaler.pkl",
        "label": "Random Forest"
    }
}

# ----------------------
# Styling (Dark Theme)
# ----------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    color: white;
}
[data-testid="stHeader"] {
    background-color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Sidebar Controls
# ----------------------
st.sidebar.title("⚙️ Controls")

# Dynamic stock list from MODEL_CONFIG
stock = st.sidebar.selectbox("Select Stock", list(MODEL_CONFIG.keys()))

initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)

# Show which ML model is used for this stock
st.sidebar.markdown("### 🧠 Model Used")
st.sidebar.info(MODEL_CONFIG[stock]["label"])

# ----------------------
# Main UI
# ----------------------
st.title("📈 ML-Assisted Quantitative Trading System")
st.markdown(f"""
### Machine Learning Based Paper Trading Simulator  
Uses Technical Indicators + **Stock-Specific ML Models** (Ensemble / RF / XGBoost)  
Currently selected stock: **{stock}**
""")

# ----------------------
# Load Models & Scalers (DYNAMIC)
# ----------------------
config = MODEL_CONFIG[stock]
model = joblib.load(config["model_path"])
scaler = joblib.load(config["scaler_path"])
model_type = config["model_type"]

# ----------------------
# Feature Engineering
# ----------------------
def add_features(df):
    features = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    features['return_1d'] = close.pct_change(1)
    features['return_5d'] = close.pct_change(5)
    features['SMA_20'] = close.rolling(20).mean()
    features['SMA_50'] = close.rolling(50).mean()
    features['EMA_20'] = close.ewm(span=20, adjust=False).mean()

    bb_std = close.rolling(20).std()
    features['BB_upper'] = features['SMA_20'] + 2 * bb_std
    features['BB_lower'] = features['SMA_20'] - 2 * bb_std

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    features['RSI'] = 100 - (100 / (1 + RS))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features['MACD'] = ema12 - ema26

    hl = high - low
    hc = abs(high - close.shift(1))
    lc = abs(low - close.shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    features["ATR"] = tr.rolling(14).mean()

    features['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    features.dropna(inplace=True)

    return features

# ----------------------
# Load Stock Data
# ----------------------
data = yf.download(stock, start="2023-01-01", end="2024-12-01", auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ----------------------
# Feature Engineering
# ----------------------
X = add_features(data)

if hasattr(scaler, "feature_names_in_"):
    # Ensure same column order as training
    X = X[scaler.feature_names_in_]

data = data.loc[X.index]
data = pd.concat([data, X], axis=1)

# ----------------------
# Scale + Predict (Safe for different model types)
# ----------------------
X_scaled = scaler.transform(X)

# If model supports predict_proba, you can use probs + threshold
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_scaled)[:, 1]
    data["Prediction"] = (probs > 0.5).astype(int)
else:
    data["Prediction"] = model.predict(X_scaled)

# ----------------------
# Paper Trading Simulation
# ----------------------
capital = initial_capital
shares = 0
position = 0
portfolio = []

trade_points = []

for i in range(len(data)):
    price = float(data.iloc[i]["Close"])
    pred = int(data.iloc[i]["Prediction"])
    sma20 = float(data.iloc[i]["SMA_20"])
    sma50 = float(data.iloc[i]["SMA_50"])

    buy_signal = (pred == 1) and (sma20 > sma50)
    sell_signal = (pred == 0) or (sma20 < sma50)

    if buy_signal and position == 0:
        shares = capital // price
        capital -= shares * price
        position = 1
        trade_points.append(("BUY", data.index[i], price))

    elif sell_signal and position == 1:
        capital += shares * price
        shares = 0
        position = 0
        trade_points.append(("SELL", data.index[i], price))

    portfolio.append(capital + shares * price)

data["Portfolio"] = portfolio

# ----------------------
# Metric Cards
# ----------------------
col1, col2, col3 = st.columns(3)

final_value = portfolio[-1]
profit = final_value - initial_capital
returns = ((final_value / initial_capital) - 1) * 100

col1.metric("💰 Final Capital", f"₹{final_value:,.2f}")
col2.metric("📈 Profit", f"₹{profit:,.2f}")
col3.metric("🔹 Return %", f"{returns:.2f}%")

# ----------------------
# Charts with Buy/Sell Markers
# ----------------------
st.subheader("📊 Trading Chart with Signals")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="Close Price")

for trade in trade_points:
    if trade[0] == "BUY":
        ax.scatter(trade[1], trade[2], color="green", marker="^", s=100)
    else:
        ax.scatter(trade[1], trade[2], color="red", marker="v", s=100)

ax.set_title(f"{stock} - Trading Signals ({MODEL_CONFIG[stock]['label']})")
ax.legend()
st.pyplot(fig)

# ----------------------
# Additional Charts
# ----------------------
st.subheader("📈 Stock Price")
st.line_chart(data["Close"])

st.subheader("📉 Portfolio Performance")
st.line_chart(data["Portfolio"])

# ----------------------
# Prediction Table
# ----------------------
st.subheader("🧠 Model Predictions (Last 20 Days)")
st.dataframe(data[["Close", "Prediction"]].tail(20))

# ----------------------
# Strategy Explanation
# ----------------------
st.markdown(f"""
### ⚙️ Trading Strategy Logic

**Stock-Specific Model:**
- Current stock: **{stock}**
- Model used: **{MODEL_CONFIG[stock]['label']}**

**BUY Signal:**
- ML Prediction = 1  
- SMA 20 > SMA 50  

**SELL Signal:**
- ML Prediction = 0  
- SMA 20 < SMA 50  

This combines **Machine Learning + Technical Analysis** for intelligent trading decisions.

---
Developed as a Mini Project on *ML-Based Quantitative Trading Systems*.
""")
