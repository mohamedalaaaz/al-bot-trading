
# ============================================================
# 🚀 BTC AI TRADING BOT — ALL IN ONE FILE
# Live Trading + AI + Dashboard (Streamlit)
# ============================================================

import pandas as pd
import numpy as np
import requests, time, hmac, hashlib, json
import websocket
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import threading

# ================= CONFIG =================
SYMBOL = "BTCUSDT"
API_KEY = "YOUR_API_KEY"
SECRET  = "YOUR_SECRET_KEY"
BALANCE = 1000
RISK = 0.01

df = pd.DataFrame()
model = None

# ================= FEATURES =================
def build_features(df):
    d = df.copy()
    d["body"] = d["close"] - d["open"]
    d["body_pct"] = d["body"] / d["open"] * 100

    d["delta"] = d["volume"] * np.sign(d["body"])
    d["delta_pct"] = d["delta"] / d["volume"]

    d["vol_z"] = (d["volume"] - d["volume"].rolling(20).mean()) / d["volume"].rolling(20).std()

    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap20"] = (tp * d["volume"]).rolling(20).sum() / d["volume"].rolling(20).sum()
    d["vwap20_dev"] = (d["close"] - d["vwap20"]) / d["vwap20"] * 100

    d["cvd"] = d["delta"].cumsum()
    d["cvd_slope3"] = d["cvd"].diff(3)

    # Liquidity sweeps
    d["sweep_high"] = (d["high"] > d["high"].rolling(10).max().shift(1)) & (d["close"] < d["high"].shift(1))
    d["sweep_low"]  = (d["low"] < d["low"].rolling(10).min().shift(1)) & (d["close"] > d["low"].shift(1))

    return d.dropna()

# ================= AI MODEL =================
def train_model(df):
    features = df[["delta_pct","vol_z","vwap20_dev","cvd_slope3","body_pct"]]
    target = (df["close"].shift(-3) > df["close"]).astype(int)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(features.dropna(), target.dropna())
    return model

def predict(model, row):
    X = [[row["delta_pct"],row["vol_z"],row["vwap20_dev"],row["cvd_slope3"],row["body_pct"]]]
    return model.predict_proba(X)[0][1]

# ================= STRATEGY =================
def score(row):
    s = 0
    if row["sweep_low"]: s += 3
    if row["sweep_high"]: s -= 3
    if row["cvd_slope3"] > 0: s += 1
    else: s -= 1
    if row["vwap20_dev"] < -0.3: s += 1
    if row["vwap20_dev"] > 0.3: s -= 1
    return s

def decision(row, model):
    s = score(row)
    p = predict(model, row)

    if s >= 3 and p > 0.6:
        return "BUY", p
    elif s <= -3 and p < 0.4:
        return "SELL", p
    return "WAIT", p

# ================= EXECUTION =================
def sign(params):
    query = "&".join([f"{k}={v}" for k,v in params.items()])
    return hmac.new(SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def place_order(side, qty):
    url = "https://fapi.binance.com/fapi/v1/order"
    params = {
        "symbol": SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "timestamp": int(time.time()*1000)
    }
    params["signature"] = sign(params)
    headers = {"X-MBX-APIKEY": API_KEY}
    requests.post(url, params=params, headers=headers)

# ================= RISK =================
def position_size(price, stop):
    risk_amt = BALANCE * RISK
    return round(risk_amt / abs(price - stop), 3)

# ================= DASHBOARD =================
st.set_page_config(layout="wide")
st.title("🚀 BTC AI Trading Bot")

price_box = st.empty()
signal_box = st.empty()
prob_box = st.empty()

# ================= PIPELINE =================
def process_candle(candle):
    global df, model

    df = pd.concat([df, pd.DataFrame([candle])]).tail(300)

    df_feat = build_features(df)

    if len(df_feat) < 50:
        return

    if model is None:
        model = train_model(df_feat)

    row = df_feat.iloc[-1]
    signal, prob = decision(row, model)

    price = row["close"]

    if signal != "WAIT":
        stop = price * 0.998 if signal=="BUY" else price * 1.002
        qty = position_size(price, stop)
        # place_order(signal, qty)  # UNCOMMENT TO TRADE LIVE

    # update UI
    price_box.metric("Price", price)
    signal_box.metric("Signal", signal)
    prob_box.metric("AI Confidence", f"{prob*100:.1f}%")

# ================= WEBSOCKET =================
def start_ws():
    def on_message(ws, msg):
        data = json.loads(msg)
        k = data["k"]

        if k["x"]:
            candle = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"])
            }
            process_candle(candle)

    ws = websocket.WebSocketApp(
        "wss://fstream.binance.com/ws/btcusdt@kline_1m",
        on_message=on_message
    )
    ws.run_forever()

# ================= RUN =================
threading.Thread(target=start_ws).start()
