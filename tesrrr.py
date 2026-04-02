# ============================================================
# 🚀 ADVANCED AI CRYPTO TRADING BOT (PRO VERSION)
# ============================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd

import ta

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from binance.client import Client
from binance import ThreadedWebsocketManager

import streamlit as st
import threading

# ================= CONFIG =================
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"

SYMBOL = "BTCUSDT"
INTERVAL = "1m"

TRADE = False   # ⚠️ set True for real trading
BALANCE = 1000
RISK = 0.01

client = Client(API_KEY, API_SECRET)

df = pd.DataFrame()

scaler = MinMaxScaler()
model = None


# ================= FEATURES =================
def add_indicators(data):
    d = data.copy()

    d["rsi"] = ta.momentum.RSIIndicator(d["close"], 14).rsi()
    d["ema20"] = ta.trend.EMAIndicator(d["close"], 20).ema_indicator()
    d["ema50"] = ta.trend.EMAIndicator(d["close"], 50).ema_indicator()

    d["macd"] = ta.trend.MACD(d["close"]).macd()
    d["volatility"] = d["high"] - d["low"]

    d = d.dropna()
    return d


# ================= AI MODEL =================
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def prepare_data(data):
    features = data[["close", "rsi", "ema20", "ema50", "macd", "volatility"]]

    scaled = scaler.fit_transform(features)

    X, y = [], []

    for i in range(50, len(scaled)):
        X.append(scaled[i-50:i])
        y.append(1 if data["close"].iloc[i] > data["close"].iloc[i-1] else 0)

    return np.array(X), np.array(y)


def train_ai(data):
    global model

    X, y = prepare_data(data)

    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    return model


def predict_next(data):
    global model

    features = data[["close", "rsi", "ema20", "ema50", "macd", "volatility"]]
    scaled = scaler.transform(features)

    X = np.array([scaled[-50:]])

    pred = model.predict(X, verbose=0)[0][0]

    return pred


# ================= RISK =================
def position_size(price):
    risk_amount = BALANCE * RISK
    return round(risk_amount / price, 3)


# ================= TRADE EXECUTION =================
def place_order(side, qty):
    try:
        order = client.create_order(
            symbol=SYMBOL,
            side=side,
            type="MARKET",
            quantity=qty
        )
        print("ORDER:", order)
    except Exception as e:
        print("ORDER ERROR:", e)


# ================= STRATEGY =================
def strategy(df):
    global model

    if len(df) < 100:
        return "WAIT", 0

    df = add_indicators(df)

    if model is None:
        train_ai(df)

    prob = predict_next(df)

    last = df.iloc[-1]

    signal = "WAIT"

    # AI + EMA trend filter
    if prob > 0.65 and last["ema20"] > last["ema50"]:
        signal = "BUY"

    elif prob < 0.35 and last["ema20"] < last["ema50"]:
        signal = "SELL"

    return signal, prob


# ================= PIPELINE =================
def process_candle(candle):
    global df

    df = pd.concat([df, pd.DataFrame([candle])]).tail(500)

    df_ind = add_indicators(df)

    signal, prob = strategy(df_ind)

    price = df_ind["close"].iloc[-1]

    if signal != "WAIT":
        qty = position_size(price)

        if TRADE:
            place_order(signal, qty)

    # dashboard output
    print(f"Price: {price} | Signal: {signal} | AI: {prob:.2f}")


# ================= WEBSOCKET =================
def start_ws():
    def handle(msg):
        k = msg["k"]

        if k["x"]:
            candle = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"])
            }

            process_candle(candle)

    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()

    twm.start_kline_socket(callback=handle, symbol=SYMBOL, interval=INTERVAL)
    twm.join()


# ================= RUN =================
if __name__ == "__main__":
    start_ws()


# ============================================================
# 🚀 PRO MAX AI CRYPTO TRADING BOT
# ============================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd

import ta
import threading
import requests

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from binance.client import Client
from binance import ThreadedWebsocketManager


# ================= CONFIG =================
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1m"

TRADE = False  # ⚠️ set True for real trading
BALANCE = 1000
RISK = 0.01

TELEGRAM = False
TG_TOKEN = "YOUR_TOKEN"
TG_CHAT = "YOUR_CHAT_ID"

client = Client(API_KEY, API_SECRET)

data_store = {sym: pd.DataFrame() for sym in SYMBOLS}
models = {}
scaler = MinMaxScaler()


# ================= TELEGRAM =================
def send_telegram(msg):
    if not TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TG_CHAT, "text": msg})


# ================= INDICATORS =================
def add_features(df):
    d = df.copy()

    d["rsi"] = ta.momentum.RSIIndicator(d["close"], 14).rsi()
    d["ema20"] = ta.trend.EMAIndicator(d["close"], 20).ema_indicator()
    d["ema50"] = ta.trend.EMAIndicator(d["close"], 50).ema_indicator()

    d["macd"] = ta.trend.MACD(d["close"]).macd()
    d["volatility"] = d["high"] - d["low"]

    # Smart Money Concepts
    d["liq_high"] = d["high"].rolling(10).max()
    d["liq_low"] = d["low"].rolling(10).min()

    d["sweep_high"] = (d["high"] > d["liq_high"].shift(1)) & (d["close"] < d["liq_high"].shift(1))
    d["sweep_low"] = (d["low"] < d["liq_low"].shift(1)) & (d["close"] > d["liq_low"].shift(1))

    return d.dropna()


# ================= AI MODEL =================
def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(50, 6)),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def prepare(df):
    features = df[["close","rsi","ema20","ema50","macd","volatility"]]
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(50, len(scaled)):
        X.append(scaled[i-50:i])
        y.append(1 if df["close"].iloc[i] > df["close"].iloc[i-1] else 0)

    return np.array(X), np.array(y)


def train(df):
    model = build_model()
    X, y = prepare(df)
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    return model


def predict(model, df):
    features = df[["close","rsi","ema20","ema50","macd","volatility"]]
    scaled = scaler.transform(features)
    X = np.array([scaled[-50:]])
    return model.predict(X, verbose=0)[0][0]


# ================= STRATEGY (PRO MAX) =================
def signal_engine(df, model):
    df = add_features(df)

    if len(df) < 100:
        return "WAIT", 0

    if model is None:
        model = train(df)

    prob = predict(model, df)

    last = df.iloc[-1]

    score = 0

    # AI
    if prob > 0.65: score += 2
    if prob < 0.35: score -= 2

    # Trend filter
    if last["ema20"] > last["ema50"]: score += 1
    else: score -= 1

    # Smart money
    if last["sweep_low"]: score += 2
    if last["sweep_high"]: score -= 2

    signal = "WAIT"
    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"

    return signal, prob


# ================= RISK =================
def size(price):
    risk_amt = BALANCE * RISK
    return round(risk_amt / price, 3)


# ================= EXECUTION =================
def order(symbol, side, qty):
    try:
        client.create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty
        )
    except Exception as e:
        print("ORDER ERROR:", e)


# ================= PIPELINE =================
def process(symbol, candle):
    global data_store, models

    df = data_store[symbol]
    df = pd.concat([df, pd.DataFrame([candle])]).tail(500)
    data_store[symbol] = df

    df_feat = add_features(df)

    if symbol not in models:
        models[symbol] = train(df_feat)

    model = models[symbol]

    signal, prob = signal_engine(df_feat, model)

    price = df_feat["close"].iloc[-1]

    if signal != "WAIT":
        qty = size(price)

        if TRADE:
            order(symbol, signal, qty)

        msg = f"{symbol} | {signal} | {prob:.2f} | {price}"
        print(msg)
        send_telegram(msg)


# ================= WEBSOCKET =================
def start():
    def handle(msg):
        k = msg["k"]
        if k["x"]:
            candle = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"])
            }
            symbol = msg["s"]
            process(symbol, candle)

    twm = ThreadedWebsocketManager(API_KEY, API_SECRET)
    twm.start()

    for sym in SYMBOLS:
        twm.start_kline_socket(callback=handle, symbol=sym, interval=INTERVAL)

    twm.join()


# ================= RUN =================
if __name__ == "__main__":
    start()