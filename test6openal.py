# elite_institutional_bot.py

import pandas as pd
import numpy as np
import requests, time, hmac, hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# =========================
# CONFIG
# =========================
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BASE_URL = "https://fapi.binance.com"
mode = "paper"   # change to "live" later

balance = 1000
risk_pct = 0.01

API_KEY = "1130043722"
SECRET  = ""

# =========================
# DATA
# =========================
def fetch(symbol):
    r = requests.get(f"{BASE_URL}/fapi/v1/klines",
                     params={"symbol":symbol,"interval":"5m","limit":200}).json()
    df = pd.DataFrame(r)
    df.columns = ["time","o","h","l","c","v","ct","q","n","tb","tq","i"]
    df["c"] = df["c"].astype(float)
    df["v"] = df["v"].astype(float)
    return df

# =========================
# FEATURES
# =========================
def features(df):
    df["ret"] = df["c"].pct_change()
    df["ma"]  = df["c"].rolling(20).mean()
    df["mom"] = df["c"] - df["c"].shift(5)
    df["vol"] = df["v"].rolling(20).mean()
    return df.dropna()

# =========================
# LABELS
# =========================
def label(df):
    df["f"] = df["c"].shift(-3)/df["c"] - 1
    df["target"] = (df["f"] > 0.002).astype(int)
    return df.dropna()

# =========================
# AI MODELS
# =========================
def train_models(df):
    df = label(df)
    X = df[["ret","ma","mom","vol"]]
    y = df["target"]

    models = {
        "rf": RandomForestClassifier(n_estimators=150),
        "xgb": XGBClassifier(n_estimators=150),
        "nn": MLPClassifier(hidden_layer_sizes=(64,32))
    }

    for m in models.values():
        m.fit(X, y)

    return models

def ensemble_predict(models, df):
    X = df[["ret","ma","mom","vol"]].iloc[-1:]

    preds = [
        models["rf"].predict_proba(X)[0][1],
        models["xgb"].predict_proba(X)[0][1],
        models["nn"].predict_proba(X)[0][1]
    ]

    return np.mean(preds)

# =========================
# ORDER BOOK (REAL)
# =========================
def get_order_book(symbol):
    data = requests.get(
        f"{BASE_URL}/fapi/v1/depth",
        params={"symbol":symbol,"limit":50}
    ).json()

    bids = np.array(data["bids"], dtype=float)
    asks = np.array(data["asks"], dtype=float)

    return bids, asks

def liquidity_signal(bids, asks):
    bid_vol = bids[:,1].sum()
    ask_vol = asks[:,1].sum()

    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    return imbalance

# =========================
# MATH FILTER (LIGHT VERSION)
# =========================
def math_filter(df, imbalance):
    score = 0

    # Trend
    if df["c"].iloc[-1] > df["ma"].iloc[-1]:
        score += 1
    else:
        score -= 1

    # Volatility
    vol = df["ret"].std()
    if vol < 0.01:
        score += 1
    else:
        score -= 1

    # Liquidity
    if imbalance > 0.1:
        score += 1
    elif imbalance < -0.1:
        score -= 1

    return score

# =========================
# PORTFOLIO
# =========================
def correlation_matrix(data):
    returns = pd.DataFrame({
        k: v["c"].pct_change() for k,v in data.items()
    })
    return returns.corr()

def allocate(signals, corr):
    weights = {}

    for sym, score in signals.items():
        base = abs(score)
        penalty = corr[sym].mean()
        weights[sym] = base * (1 - penalty)

    total = sum(weights.values())
    for k in weights:
        weights[k] /= total if total > 0 else 1

    return weights

# =========================
# EXECUTION
# =========================
def sign(params):
    qs = "&".join([f"{k}={v}" for k,v in params.items()])
    return hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()

def place_order(symbol, side, qty):
    if mode == "paper":
        print(f"[PAPER] {side} {symbol} {qty}")
        return

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "timestamp": int(time.time()*1000)
    }

    params["signature"] = sign(params)

    headers = {"X-MBX-APIKEY": API_KEY}
    r = requests.post(BASE_URL+"/fapi/v1/order",
                      params=params, headers=headers)
    print(r.json())

# =========================
# MAIN LOOP
# =========================
data_store = {}

while True:

    signals = {}

    for sym in SYMBOLS:

        df = fetch(sym)
        df = features(df)

        models = train_models(df)
        prob = ensemble_predict(models, df)

        bids, asks = get_order_book(sym)
        imbalance = liquidity_signal(bids, asks)

        math = math_filter(df, imbalance)

        score = 0
        if prob > 0.6: score += 2
        elif prob < 0.4: score -= 2

        score += math

        signals[sym] = score
        data_store[sym] = df

    corr = correlation_matrix(data_store)
    weights = allocate(signals, corr)

    for sym in signals:
        price = data_store[sym]["c"].iloc[-1]
        qty = (balance * weights[sym] * risk_pct) / price

        if signals[sym] > 2:
            place_order(sym, "BUY", round(qty, 3))

        elif signals[sym] < -2:
            place_order(sym, "SELL", round(qty, 3))

    time.sleep(15)