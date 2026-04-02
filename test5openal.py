# ===== CORE =====
import os
import time
import json
import asyncio
import threading
import datetime as dt

# ===== DATA HANDLING =====
import numpy as np
import pandas as pd

# ===== TECHNICAL INDICATORS =====
import ta  # pip install ta

# ===== MACHINE LEARNING / AI =====
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===== EXCHANGE API (BINANCE) =====
from binance.client import Client
from binance import ThreadedWebsocketManager

# ===== VISUALIZATION / DASHBOARD =====
import dash
from dash import dcc, html
import plotly.graph_objs as go

# ===== LOGGING =====
import logging

# ===== WARNINGS CONTROL =====
import warnings
warnings.filterwarnings("ignore")







# === Config and Globals ===
SYMBOL       = "BTCUSDT"
BASE_URL     = "https://fapi.binance.com"
IMBALANCE_THR= 3.0
BIG_TRADE_X  = 5.0
LIQUIDITY_ATR= 1.5
API_KEY      = "<your key>"
API_SECRET   = "<your secret>"
BALANCE      = 10000  # example

data_1m = pd.DataFrame()
model = None

# === Feature Engineering ===
def build_base(df):
    d = df.copy()
    d["body"] = d["close"] - d["open"]
    d["body_pct"] = d["body"] / d["open"] * 100
    d["is_bull"] = d["body"] > 0
    d["range"] = d["high"] - d["low"]
    d["range_pct"] = d["range"] / d["open"] * 100
    d["wick_top"] = d["high"] - d[["open","close"]].max(axis=1)
    d["wick_bot"] = d[["open","close"]].min(axis=1) - d["low"]
    d["sell_vol"] = d["volume"] - d["taker_buy_vol"]
    d["delta"] = d["taker_buy_vol"] - d["sell_vol"]
    d["delta_pct"] = d["delta"] / d["volume"].replace(0, np.nan)
    # ATR (14)
    hl = d["high"] - d["low"]
    hpc = (d["high"] - d["close"].shift(1)).abs()
    lpc = (d["low"] - d["close"].shift(1)).abs()
    d["atr"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / d["volume"].rolling(50).std()
    d["hour"] = d["open_time"].dt.hour
    # session mapping (same as user code)
    d["session"] = d["hour"].apply(lambda h: "Asia" if 0<=h<8 else "London" if 8<=h<13 else "NY" if 13<=h<20 else "Late")
    return d.dropna().reset_index(drop=True)
















# Integrate user’s modules (CVD Pro, Big Traders, etc.)
def cvd_pro(df):
    d = df.copy()
    d["cvd"] = d["delta"].cumsum()
    d["cvd_roll20"] = d["delta"].rolling(20).sum()
    d["cvd_slope3"] = d["cvd_roll20"].diff(3)
    d["price_slope3"] = d["close"].pct_change(3) * 100
    d["delta_exhaust"] = ((d["is_bull"] & (d["delta_pct"]<0.05)) |
                         (~d["is_bull"] & (d["delta_pct"]>-0.05)))
    d["div_bull"] = (d["price_slope3"] < -0.15) & (d["cvd_slope3"] > 0)
    d["div_bear"] = (d["price_slope3"] >  0.15) & (d["cvd_slope3"] < 0)
    d["absorption"] = (d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.1)
    return d

def big_traders(df):
    d = df.copy()
    d["is_big"] = d["vol_z"] > BIG_TRADE_X
    d["is_moderate"] = (d["vol_z"] > 2.5) & (d["vol_z"] <= BIG_TRADE_X)
    avg_trade = d["volume"] / d["trades"].replace(0,np.nan)
    global_avg = avg_trade.median()
    d["avg_trade_size"] = avg_trade
    d["iceberg_prob"] = (
        (d["vol_z"] > 1.5) &
        (d["trades"] > d["trades"].rolling(50).mean()*1.5) &
        (avg_trade < global_avg * 0.7)
    )
    d["spoof_suspect"] = (
        (d["is_bull"] & (d["delta_pct"] < -0.3)) |
        (~d["is_bull"] & (d["delta_pct"] > 0.3))
    )
    d["accum_signal"] = ((~d["is_bull"]) & (d["delta_pct"] > 0.15) & (d["vol_z"] > 0.5))
    d["distrib_signal"] = ((d["is_bull"]) & (d["delta_pct"] < -0.15) & (d["vol_z"] > 0.5))
    return d

# (Other modules OrderFlow, UnfinishedBusiness, VWAP/TWAP, etc. implemented similarly)





# === AI Model Training & Persistence ===
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(historic_df):
    # Prepare features and binary target (e.g. 1 if price rises in next 3 candles)
    df_feat = build_base(historic_df)
    df_feat = pd.concat([cvd_pro(df_feat), big_traders(df_feat), order_flow(df_feat),
                         unfinished_business(df_feat)[0], vwap_twap(df_feat)], axis=1)  # sample merge
    df_feat.dropna(inplace=True)
    features = df_feat[["delta_pct","vol_z","vwap20_dev","cvd_slope3","body_pct"]]
    target = (df_feat["close"].shift(-3) > df_feat["close"]).astype(int)
    # Split and train
    X, y = features[:-3], target[:-3]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "btc_model.joblib")
    return model

def load_model():
    try:
        return joblib.load("btc_model.joblib")
    except:
        return train_model(data_1m)  # fallback: train on initial data
# === Signal Engine (Combining AI + Rules) ===
def compute_trade_signal(feature_df, model):
    """Return (signal, ai_prob)."""
    last = feature_df.iloc[-1]
    score = 0
    # Rule-based scoring (preserve thresholds)
    if last["div_bull"]:    score += 2
    if last["div_bear"]:    score -= 2
    if last["vwap20_dev"] < -0.3: score += 1
    if last["vwap20_dev"] > 0.3:  score -= 1
    if last["vol_z"] > 2.5:
        score += (1 if last["is_bull"] else -1)
    # More rules can be added from other modules...
    # Get AI probability
    feat = [[last["delta_pct"], last["vol_z"], last["vwap20_dev"], last["cvd_slope3"], last["body_pct"]]]
    prob_up = model.predict_proba(feat)[0][1]
    # Decision thresholds
    if score >= 3 and prob_up > 0.6:
        return "BUY", prob_up
    if score <= -3 and prob_up < 0.4:
        return "SELL", prob_up
    return "HOLD", prob_up
# === Execution & Risk ===
import time
def sign_params(params):
    query = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return signature

def place_order(side, qty):
    url = f"{BASE_URL}/fapi/v1/order"
    params = {"symbol": SYMBOL, "side": side, "type": "MARKET",
              "quantity": qty, "timestamp": int(time.time()*1000)}
    params["signature"] = sign_params(params)
    headers = {"X-MBX-APIKEY": API_KEY}
    res = requests.post(url, params=params, headers=headers)
    return res.json()

def position_size(entry, stop):
    risk_amount = BALANCE * 0.01
    size = risk_amount / abs(entry - stop)
    return round(size, 6)
# === Streamlit Dashboard Setup ===
st.set_page_config(layout="wide")
st.title("BTC Institutional Trading Bot")

price_box  = st.empty()
signal_box = st.empty()
prob_box   = st.empty()

def update_ui(current_price, signal, prob):
    price_box.metric("Price (BTCUSDT)", f"{current_price:.0f}")
    signal_box.metric("Signal", signal)
    prob_box.metric("AI Win Prob", f"{prob*100:.1f}%")
    # (Additional charts can be drawn via st.line_chart, etc.)
# === WebSocket Handler ===
def on_message(ws, msg):
    global data_1m, model
    data = json.loads(msg)
    k = data["k"]
    if k["x"]:  # candle closed
        candle = {"open_time": pd.to_datetime(k["t"], unit='ms'),
                  "open": float(k["o"]), "high": float(k["h"]),
                  "low": float(k["l"]), "close": float(k["c"]),
                  "volume": float(k["v"]), "taker_buy_vol": float(k["q"]), "trades": int(k["n"])}
        data_1m = pd.concat([data_1m, pd.DataFrame([candle])]).drop_duplicates("open_time")
        data_1m.sort_values("open_time", inplace=True)
        # Build features
        df_feat = build_base(data_1m)
        df_feat = cvd_pro(df_feat)
        df_feat = big_traders(df_feat)
        # ... apply other modules similarly ...
        # Ensure model is loaded
        if model is None:
            model = load_model()
        # Compute signal
        signal, prob = compute_trade_signal(df_feat, model)
        price = df_feat["close"].iloc[-1]
        # Execute trade
        if signal == "BUY":
            stop = price * 0.998
            qty = position_size(price, stop)
            place_order("BUY", qty)
        elif signal == "SELL":
            stop = price * 1.002
            qty = position_size(price, stop)
            place_order("SELL", qty)
        # Update dashboard
        update_ui(price, signal, prob)

def run_ws():
    ws = websocket(f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@kline_1m",
                      on_message=on_message)
    ws.run_forever()

def test_build_base():
    df = pd.DataFrame([{"open_time": pd.Timestamp.utcnow(), "open":10, "high":12, "low":9, "close":11, "volume":100, "taker_buy_vol":60, "trades":200}])
    out = build_base(df)
    assert np.isclose(out["body_pct"].iloc[0], (11-10)/10*100)
    assert out["is_bull"].iloc[0] == True


# === Main Entry ===
if __name__ == "__main__":
    # Initialize historical data (if needed)
    # Train or load ML model
    model = train_model(data_1m)  # first run, then saved
    # Start WebSocket thread
    threading.Thread(target=run_ws).start()
