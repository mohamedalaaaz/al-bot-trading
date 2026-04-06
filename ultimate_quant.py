#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ULTIMATE QUANT ENGINE v5.0  —  COMPLETE MAIN ENGINE                     ║
║   Run: python ultimate_quant.py --paper                                    ║
║   TWO FILES REQUIRED: uq_math.py + ultimate_quant.py (same folder)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

FEATURES:
  • Model persistence: saves/loads full trained state to disk (pickle + numpy)
  • Online learning: SGD updates every bar, Bayesian posteriors after every trade
  • Concept drift detection (Page-Hinkley) → auto-retrain
  • Bayesian model averaging: GBM + ResNet + ET combined with posterior weights
  • Particle filter + RTS Kalman smoother for nonlinear state estimation
  • EVT (GEV + GPD) for tail risk and dynamic position sizing
  • Information theory: transfer entropy, mutual information, variance ratio
  • Copula dependency: Gaussian + Clayton, tail dependence, effective N signals
  • Heston stochastic volatility model
  • Advanced Kelly: uncertainty-adjusted + CVaR-constrained + Sharpe-optimal
  • Triple-barrier labeling (ATR-adaptive, Lopez de Prado)
  • Purged K-Fold + embargo (no leakage)
  • Fractional differentiation
  • Meta-labeling with isotonic calibration
  • Deep ResNet (skip connections, Swish, Adam)
  • CPCV Sharpe estimation (combinatorial purged CV)
  • WebSocket real-time: kline + aggTrade streams with auto-reconnect
  • Paper trading with TP1/TP2/SL management and live P&L
  • Signal history with live accuracy tracking

INSTALL:
  pip install requests pandas numpy scipy scikit-learn websocket-client

RUN:
  python ultimate_quant.py                 # load saved model or train fresh
  python ultimate_quant.py --paper         # paper trading mode
  python ultimate_quant.py --account 5000  # set account size USDT
  python ultimate_quant.py --tf 1m         # timeframe (1m/3m/5m/15m)
  python ultimate_quant.py --reset         # delete saved model, train fresh
"""

import os, sys, math, time, json, pickle, random, warnings, argparse, threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.signal import hilbert as sp_hilbert
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

# ── Import advanced math suite (uq_math.py) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from uq_math import (AdvancedMathSuite, BayesianEngine, AdvancedKelly,
                          OnlineLearner, EVTEngine, InfoTheoryEngine,
                          StochasticEngine, CopulaEngine)
    MATH_OK = True
    print("  [OK] uq_math.py loaded — advanced statistics active")
except ImportError as _e:
    MATH_OK = False
    print("  [WARN] uq_math.py not found: {}".format(_e))
    print("  [WARN] Place uq_math.py in same folder. Advanced math disabled.")

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import requests
    NET = True
except ImportError:
    NET = False

try:
    import websocket as ws_module
    WS_OK = True
except ImportError:
    WS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "SYMBOL":        "BTCUSDT",
    "TF":            "5m",
    "CANDLES":       500,
    "ACCOUNT":       1000.0,
    "MAX_RISK":      0.015,
    "MIN_SCORE":     6,
    "MIN_CONF":      55.0,
    "MIN_META":      0.52,
    "MIN_RR":        1.5,
    "ATR_SL":        1.5,
    "TP_MULT":       2.5,
    "TARGET_BARS":   5,
    "BARRIER_PCT":   0.008,
    "FRAC_D":        0.40,
    "PCA_VAR":       0.90,
    "PURGE":         5,
    "EMBARGO":       2,
    "GBM_N":         300,
    "ET_N":          200,
    "NN_H":          64,
    "NN_B":          3,
    "NN_EP":         100,
    "NN_LR":         5e-4,
    "NN_L2":         1e-4,
    "NN_DR":         0.25,
    "MODEL_DIR":     "uq_models",
    "CHECKPOINT_N":  30,
    "RETRAIN_N":     100,
    "ONLINE_N":      5,
    "PAPER_SLIP":    0.0005,
    "TICK_WIN_MS":   30000,
}

BASE_API = "https://fapi.binance.com"
BASE_WS  = "wss://fstream.binance.com/ws"

COLORS = {
    "G":"\033[92m","R":"\033[91m","Y":"\033[93m","C":"\033[96m",
    "W":"\033[97m","B":"\033[1m","D":"\033[2m","M":"\033[95m","X":"\033[0m",
}
def cc(t, col):
    return COLORS.get(col, "") + str(t) + COLORS["X"]
def bbar(v, w=10):
    n = min(int(abs(float(v)) * w), w)
    return "█" * n + "░" * (w - n)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    """Save/load entire trained model state. Supports versioned checkpoints."""

    def __init__(self, model_dir=CFG["MODEL_DIR"]):
        self.model_dir   = model_dir
        self.best_sharpe = -np.inf
        os.makedirs(model_dir, exist_ok=True)
        self.latest_path = os.path.join(model_dir, "checkpoint_latest.pkl")
        self.best_path   = os.path.join(model_dir, "checkpoint_best.pkl")
        self.meta_path   = os.path.join(model_dir, "meta.json")

    def save(self, state: dict, sharpe: float = None, tag: str = "latest") -> bool:
        try:
            path = os.path.join(self.model_dir, "checkpoint_{}.pkl".format(tag))
            # Separate large numpy arrays for efficiency
            np_data = {}
            state_small = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray) and v.nbytes > 10000:
                    np_data["arr_" + k] = v
                elif k == "resnet_weights" and isinstance(v, dict):
                    for wk, wv in v.items():
                        np_data["rn_" + wk] = wv
                else:
                    state_small[k] = v

            with open(path, "wb") as f:
                pickle.dump(state_small, f, protocol=pickle.HIGHEST_PROTOCOL)

            if np_data:
                np.savez_compressed(path.replace(".pkl", "_numpy.npz"), **np_data)

            # Update best if improved
            is_best = False
            if sharpe is not None and sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                import shutil; shutil.copy(path, self.best_path)
                np_src = path.replace(".pkl", "_numpy.npz")
                np_dst = self.best_path.replace(".pkl", "_numpy.npz")
                if os.path.exists(np_src): shutil.copy(np_src, np_dst)
                is_best = True

            # Copy to latest
            if path != self.latest_path:
                import shutil; shutil.copy(path, self.latest_path)
                np_src = path.replace(".pkl", "_numpy.npz")
                np_dst = self.latest_path.replace(".pkl", "_numpy.npz")
                if os.path.exists(np_src) and np_src != np_dst:
                    shutil.copy(np_src, np_dst)

            meta = {
                "saved_at":    datetime.now(timezone.utc).isoformat(),
                "cpcv_sharpe": float(sharpe) if sharpe else 0.0,
                "best_sharpe": float(self.best_sharpe),
                "is_best":     is_best,
                "n_samples":   state_small.get("n_samples", 0),
                "tag":         "BEST" if is_best else tag,
            }
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            return True
        except Exception as e:
            print("  [SAVE ERROR] {}".format(e))
            return False

    def load(self, prefer_best=True) -> dict:
        paths = ([self.best_path, self.latest_path] if prefer_best
                 else [self.latest_path, self.best_path])
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                # Restore numpy arrays
                np_path = path.replace(".pkl", "_numpy.npz")
                if os.path.exists(np_path):
                    np_data = np.load(np_path, allow_pickle=True)
                    rw = {}
                    for k in np_data.files:
                        if k.startswith("rn_"):
                            rw[k[3:]] = np_data[k]
                        elif k.startswith("arr_"):
                            state[k[4:]] = np_data[k]
                    if rw:
                        state["resnet_weights"] = rw
                # Print meta
                if os.path.exists(self.meta_path):
                    with open(self.meta_path) as f:
                        meta = json.load(f)
                    print("  [LOAD] {} | CPCV Sharpe={:.3f} | Samples={}".format(
                        path, meta.get("cpcv_sharpe", 0), meta.get("n_samples", 0)))
                return state
            except Exception as e:
                print("  [LOAD ERROR] {}: {}".format(path, e))
        return {}

    def exists(self) -> bool:
        return os.path.exists(self.latest_path) or os.path.exists(self.best_path)

    def delete(self):
        import shutil
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_sharpe = -np.inf
        print("  [STORE] All checkpoints deleted.")


# ─────────────────────────────────────────────────────────────────────────────
#  RESNET  (skip connections, Swish activation, Adam optimizer)
# ─────────────────────────────────────────────────────────────────────────────
class ResNet:
    def __init__(self, n_in, hidden=64, n_blocks=3, lr=5e-4, l2=1e-4, dropout=0.25):
        self.lr = lr; self.l2 = l2; self.dr = dropout; self.nb = n_blocks
        self.val_acc = 0.5

        def he(a, b): return np.random.randn(a, b).astype(np.float64) * math.sqrt(2.0 / a)
        self.Wi  = he(n_in, hidden); self.bi  = np.zeros(hidden, dtype=np.float64)
        self.Wr1 = [he(hidden, hidden) for _ in range(n_blocks)]
        self.br1 = [np.zeros(hidden, dtype=np.float64) for _ in range(n_blocks)]
        self.Wr2 = [he(hidden, hidden) for _ in range(n_blocks)]
        self.br2 = [np.zeros(hidden, dtype=np.float64) for _ in range(n_blocks)]
        self.Wo  = he(hidden, 1); self.bo = np.zeros(1, dtype=np.float64)
        ap = self._p()
        self.m = {k: np.zeros_like(v) for k, v in ap.items()}
        self.v = {k: np.zeros_like(v) for k, v in ap.items()}
        self.t = 0

    def _p(self):
        p = {"Wi": self.Wi, "bi": self.bi, "Wo": self.Wo, "bo": self.bo}
        for i in range(self.nb):
            p["W1_" + str(i)] = self.Wr1[i]; p["b1_" + str(i)] = self.br1[i]
            p["W2_" + str(i)] = self.Wr2[i]; p["b2_" + str(i)] = self.br2[i]
        return p

    def get_weights(self): return {k: v.copy() for k, v in self._p().items()}

    def set_weights(self, w):
        for k, v in w.items():
            if k in self._p(): self._p()[k][...] = v

    @staticmethod
    def _sw(x): return x / (1.0 + np.exp(-np.clip(x, -50, 50)))
    @staticmethod
    def _sd(x): s = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))); return s + x * s * (1.0 - s)
    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def fwd(self, X, train=True):
        ca = {}
        Z0 = X @ self.Wi + self.bi; A0 = self._sw(Z0)
        ca["X"] = X; ca["Z0"] = Z0; A = A0
        for i in range(self.nb):
            Z1 = A @ self.Wr1[i] + self.br1[i]; A1 = self._sw(Z1)
            if train and self.dr > 0:
                mk = (np.random.rand(*A1.shape) > self.dr).astype(np.float64) / (1.0 - self.dr + 1e-9)
                A1 *= mk; ca["mk_" + str(i)] = mk
            Z2 = A1 @ self.Wr2[i] + self.br2[i]; A2 = self._sw(Z2 + A)
            ca["Ai_" + str(i)] = A; ca["Z1_" + str(i)] = Z1
            ca["A1_" + str(i)] = A1; ca["Z2_" + str(i)] = Z2; A = A2
        Zo = A @ self.Wo + self.bo; Ao = self._sig(Zo)
        ca["Af"] = A; ca["Zo"] = Zo
        return Ao.ravel(), ca

    def bwd(self, y, out, ca):
        m = float(len(y)); g = {}
        dA = (out - y) / m; dZo = dA.reshape(-1, 1)
        g["Wo"] = ca["Af"].T @ dZo + self.l2 * self.Wo
        g["bo"] = dZo.sum(axis=0); dA = dZo @ self.Wo.T
        for i in reversed(range(self.nb)):
            Ai = ca["Ai_" + str(i)]; Z1 = ca["Z1_" + str(i)]
            A1 = ca["A1_" + str(i)]; Z2 = ca["Z2_" + str(i)]
            dA2 = dA * self._sd(Z2 + Ai)
            g["W2_" + str(i)] = A1.T @ dA2 + self.l2 * self.Wr2[i]
            g["b2_" + str(i)] = dA2.sum(axis=0)
            dA1 = dA2 @ self.Wr2[i].T
            if "mk_" + str(i) in ca: dA1 *= ca["mk_" + str(i)]
            dZ1 = dA1 * self._sd(Z1)
            g["W1_" + str(i)] = Ai.T @ dZ1 + self.l2 * self.Wr1[i]
            g["b1_" + str(i)] = dZ1.sum(axis=0)
            dA = dZ1 @ self.Wr1[i].T + dA2
        dZ0 = dA * self._sd(ca["Z0"])
        g["Wi"] = ca["X"].T @ dZ0 + self.l2 * self.Wi
        g["bi"] = dZ0.sum(axis=0)
        return g

    def _adam(self, g):
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8; p = self._p()
        for k, gv in g.items():
            if k not in p: continue
            self.m[k] = b1 * self.m.get(k, np.zeros_like(gv)) + (1 - b1) * gv
            self.v[k] = b2 * self.v.get(k, np.zeros_like(gv)) + (1 - b2) * gv ** 2
            mc = self.m[k] / (1 - b1 ** self.t); vc = self.v[k] / (1 - b2 ** self.t)
            p[k] -= self.lr * mc / (np.sqrt(vc) + eps)

    def fit(self, X, y, Xv=None, yv=None, epochs=100, batch=32):
        best_acc = 0.0; best_w = None; no_imp = 0
        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            for s in range(0, len(X), batch):
                Xb = X[idx[s:s + batch]]; yb = y[idx[s:s + batch]]
                if len(Xb) < 2: continue
                out, ca = self.fwd(Xb, True); g = self.bwd(yb, out, ca); self._adam(g)
            if Xv is not None and len(Xv) > 0:
                pv, _ = self.fwd(Xv, False); acc = float(((pv > 0.5) == yv).mean())
                if acc > best_acc: best_acc = acc; best_w = self.get_weights(); no_imp = 0
                else: no_imp += 1
                if no_imp >= 15: break
            if (ep + 1) % 20 == 0: self.lr *= 0.7
        if best_w: self.set_weights(best_w)
        self.val_acc = best_acc

    def predict(self, X): p, _ = self.fwd(X, False); return p


# ─────────────────────────────────────────────────────────────────────────────
#  DATA & FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def fetch_klines(symbol, tf, limit):
    r = requests.get("{}/fapi/v1/klines".format(BASE_API),
                     params={"symbol": symbol, "interval": tf, "limit": limit}, timeout=12)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "ts","o","h","l","c","v","ct","qv","n","tbv","tbqv","_"])
    df["open_time"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    for col in ["o","h","l","c","v","tbv","n"]: df[col] = df[col].astype(float)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close",
                               "v":"volume","tbv":"taker_buy_vol","n":"trades"})[
        ["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]

def fetch_funding(symbol):
    r = requests.get("{}/fapi/v1/fundingRate".format(BASE_API),
                     params={"symbol": symbol, "limit": 50}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(float), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df

def make_synthetic(n=500, seed=42, base=67000.0):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min", tz="UTC")
    price = float(base); rows = []
    for dt in dates:
        h = dt.hour; sv = 2.2 if h in [8,9,13,14,15,16] else 0.65
        mu = -0.00018 if h in [16,17,18] else 0.00012
        price = max(price * (1 + np.random.normal(mu, 0.0028 * sv)), 50000)
        hi = price * (1 + abs(np.random.normal(0, 0.002 * sv)))
        lo = price * (1 - abs(np.random.normal(0, 0.002 * sv)))
        vol = max(abs(np.random.normal(1100, 380)) * sv, 80.0)
        bsk = 0.63 if h in [8,9] else (0.36 if h in [17,18] else 0.50)
        tb = vol * float(np.clip(np.random.beta(bsk*7, (1-bsk)*7), 0.05, 0.95))
        if np.random.random() < 0.025: vol *= np.random.uniform(5, 9)
        rows.append({"open_time":dt,"open":price*(1+np.random.normal(0,0.001)),
                     "high":hi,"low":lo,"close":price,"volume":vol,
                     "taker_buy_vol":tb,"trades":int(vol/0.04)})
    df = pd.DataFrame(rows)
    fund = pd.DataFrame([{"fundingTime":dates[i],
                           "fundingRate":float(np.random.normal(0.0001,0.0003))}
                          for i in range(0, n, 96)])
    return df, fund

def prepare(df):
    d = df.copy()
    d["body"]      = d["close"] - d["open"]
    d["body_pct"]  = d["body"] / d["open"] * 100
    d["is_bull"]   = d["body"] > 0
    d["wick_top"]  = d["high"] - d[["open","close"]].max(axis=1)
    d["wick_bot"]  = d[["open","close"]].min(axis=1) - d["low"]
    d["sell_vol"]  = d["volume"] - d["taker_buy_vol"]
    d["delta"]     = d["taker_buy_vol"] - d["sell_vol"]
    d["delta_pct"] = (d["delta"] / d["volume"].replace(0, np.nan)).fillna(0)
    hl  = d["high"] - d["low"]
    hpc = (d["high"] - d["close"].shift(1)).abs()
    lpc = (d["low"]  - d["close"].shift(1)).abs()
    d["atr"]   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    rm = d["volume"].rolling(50).mean()
    rs = d["volume"].rolling(50).std().replace(0, np.nan)
    d["vol_z"] = (d["volume"] - rm) / rs
    d["hour"]  = d["open_time"].dt.hour
    d["dow"]   = d["open_time"].dt.dayofweek
    d["session"] = d["hour"].apply(
        lambda h: "Asia" if h < 8 else "London" if h < 13 else "NY" if h < 20 else "Late")
    return d.fillna(0)

def rsi_s(prices, period):
    d = prices.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def frac_diff(series, d, thresh=1e-5):
    w = [1.0]; k = 1
    while True:
        val = -w[-1] * (d - k + 1) / k
        if abs(val) < thresh: break
        w.append(val); k += 1
    w = np.array(w[::-1]); width = len(w)
    out = pd.Series(np.nan, index=series.index)
    for i in range(width - 1, len(series)):
        out.iloc[i] = float(np.dot(w, series.iloc[i - width + 1:i + 1].values))
    return out

def build_features(df, fund=None, tick_snap=None):
    d   = df.copy(); c_ = d["close"].astype(float)
    vol = d["volume"].astype(float).replace(0, np.nan)
    dp  = d["delta_pct"].astype(float); dlt = d["delta"].astype(float)
    ret = c_.pct_change(); lr = np.log(c_ / c_.shift(1)).fillna(0)
    tp_ = (d["high"] + d["low"] + c_) / 3
    atr = d["atr"].astype(float).replace(0, np.nan)
    F   = pd.DataFrame(index=d.index)

    # 1. Momentum (Fibonacci lags)
    for lag in [1,2,3,5,8,13,21,34]:
        F["mom_" + str(lag)] = c_.pct_change(lag)
    for fast, slow in [(8,21),(12,26),(5,13)]:
        F["macd_{}_{}".format(fast,slow)] = (c_.ewm(fast).mean() - c_.ewm(slow).mean()) / c_
    F["mom_acc"] = c_.pct_change(5) - c_.pct_change(5).shift(5)
    for w in [10,20,50]:
        hi_ = d["high"].rolling(w).max(); lo_ = d["low"].rolling(w).min()
        rng_ = (hi_ - lo_).replace(0, np.nan)
        F["rpos_"+str(w)] = (c_ - lo_) / rng_
        F["dhi_"+str(w)]  = (hi_ - c_) / c_ * 100
        F["dlo_"+str(w)]  = (c_ - lo_) / c_ * 100

    # 2. Mean reversion
    for w in [10,20,50,100]:
        mu_ = c_.rolling(w).mean(); sg_ = c_.rolling(w).std().replace(0, np.nan)
        F["z_"+str(w)] = (c_ - mu_) / sg_
    for p in [7,14,21]: F["rsi_"+str(p)] = rsi_s(c_, p)
    F["willr"] = ((d["high"].rolling(14).max() - c_) /
                  (d["high"].rolling(14).max() - d["low"].rolling(14).min() + 1e-9) * -100)

    # 3. Fractional differentiation
    for d_val in [0.3, 0.4, 0.5]:
        F["fd_"+str(d_val).replace(".","")] = frac_diff(c_, d_val)

    # 4. Order flow / delta
    F["delta_pct"] = dp; F["buy_ratio"] = d["taker_buy_vol"] / vol
    F["vol_imb"]   = dlt / vol
    cvd20 = dlt.rolling(20).sum()
    F["cvd_20n"]  = cvd20 / vol.rolling(20).mean()
    F["cvd_sl3"]  = cvd20.diff(3); F["cvd_sl5"] = cvd20.diff(5)
    F["cvd_acc"]  = cvd20.diff(3).diff(2)
    pr_s  = c_.diff(3) / c_.shift(3) * 100; cvd_s = cvd20.diff(3)
    F["div_bull"] = ((pr_s < -0.12) & (cvd_s > 0)).astype(float)
    F["div_bear"] = ((pr_s >  0.12) & (cvd_s < 0)).astype(float)
    F["exh_buy"]  = ((dp > 0.28) & (d["body_pct"].abs() < 0.06)).astype(float)
    F["exh_sell"] = ((dp < -0.28) & (d["body_pct"].abs() < 0.06)).astype(float)
    ky = pd.Series(np.nan, index=d.index)
    for i in range(20, len(d)):
        ri = ret.iloc[i-20:i].values; dpi = dp.iloc[i-20:i].values
        c2 = np.cov(ri, dpi) if len(ri) > 3 else np.zeros((2,2))
        ky.iloc[i] = float(c2[0,1] / (dpi.var() + 1e-12))
    F["kyle_lam"] = ky

    # 5. Volatility
    for w in [5,10,20,50]:
        F["rv_"+str(w)]   = (lr**2).rolling(w).sum()
        F["rvol_"+str(w)] = lr.rolling(w).std()
    F["pk_vol"] = np.sqrt((1/(4*math.log(2))) *
                           (np.log(d["high"]/d["low"].replace(0,np.nan))**2).rolling(20).mean())
    rv20 = (lr**2).rolling(20).sum()
    F["vov"]    = rv20.rolling(10).std() / rv20.rolling(10).mean().replace(0,np.nan)
    F["skew50"] = lr.rolling(50).apply(lambda x: float(sp_skew(x)),  raw=True)
    F["kurt50"] = lr.rolling(50).apply(lambda x: float(sp_kurt(x)), raw=True)
    F["vr_5_20"]= F["rvol_5"] / F["rvol_20"].replace(0,np.nan)

    # 6. VWAP / structure
    for w in [20,50,100]:
        vw_ = (tp_ * vol).rolling(w).sum() / vol.rolling(w).sum()
        vr_ = (vol * (tp_ - vw_)**2).rolling(w).sum() / vol.rolling(w).sum()
        vs_ = np.sqrt(vr_.replace(0, np.nan))
        F["vwap_dev_"+str(w)]  = (c_ - vw_) / vw_ * 100
        F["vwap_band_"+str(w)] = (c_ - vw_) / vs_.replace(0, np.nan)
    for sp in [8,21,50]:
        F["ema_dev_"+str(sp)] = (c_ - c_.ewm(sp).mean()) / c_ * 100
    F["ema_8_21"]  = (c_.ewm(8).mean() - c_.ewm(21).mean()) / c_ * 100
    F["ema_cross"] = (c_.ewm(8).mean() > c_.ewm(21).mean()).astype(float)

    # 7. Microstructure
    rng_ = (d["high"] - d["low"]).replace(0, np.nan)
    F["wt_rel"]   = d["wick_top"] / atr; F["wb_rel"] = d["wick_bot"] / atr
    F["wasym"]    = (d["wick_bot"] - d["wick_top"]) / atr
    F["effic"]    = d["body_pct"].abs() / (rng_ / c_ * 100).replace(0, np.nan)
    F["hl_pos"]   = (c_ - d["low"]) / rng_
    F["vol_z"]    = d["vol_z"]
    F["big_trade"]= (d["vol_z"] > 3.0).astype(float)
    F["absorb"]   = ((d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.08)).astype(float)
    F["trap"]     = ((d["body_pct"].shift(1).abs() > 0.25) &
                     (d["body_pct"] * d["body_pct"].shift(1) < 0)).astype(float)
    F["amihud"]   = (lr.abs() / vol).rolling(20).mean()

    # 8. Hilbert / Fisher
    try:
        raw_arr = c_.values.astype(float)
        x_dt    = raw_arr - np.linspace(raw_arr[0], raw_arr[-1], len(raw_arr))
        analytic= sp_hilbert(x_dt)
        F["hil_amp"]  = pd.Series(np.abs(analytic),   index=d.index) / (c_.std() + 1e-9)
        F["hil_phase"]= pd.Series(np.angle(analytic),  index=d.index)
        F["hil_freq"] = pd.Series(np.gradient(np.unwrap(np.angle(analytic))), index=d.index)
        hi10 = c_.rolling(10).max(); lo10 = c_.rolling(10).min()
        v_   = (2*(c_-lo10)/(hi10-lo10+1e-9)-1).clip(-0.999, 0.999)
        F["fisher"] = 0.5 * np.log((1+v_) / (1-v_+1e-10))
    except Exception:
        for col in ["hil_amp","hil_phase","hil_freq","fisher"]: F[col] = 0.0

    # 9. Wyckoff
    n_w = min(30, len(d)); x_w = np.arange(n_w); rec = d.tail(n_w)
    def sl_(vals):
        try:    return float(np.polyfit(x_w[:len(vals)], vals, 1)[0])
        except: return 0.0
    pt = sl_(rec["close"].values); bt = sl_(rec["taker_buy_vol"].values)
    st = sl_((rec["volume"]-rec["taker_buy_vol"]).values)
    wy = (2 if pt<-0.3 and bt>0 else 3 if pt>0.3 and bt>0 else
          -2 if pt>0.3 and st>0 else -3 if pt<-0.3 and st>0 else 0)
    F["wyckoff"] = float(wy)
    cvd_t = 0.0
    if len(d) >= 20:
        v0 = float(dlt.rolling(20).sum().iloc[-1])
        v1 = float(dlt.rolling(20).sum().iloc[-20])
        cvd_t = float(np.clip((v0-v1)/10000, -3, 3))
    F["sm_flow"] = cvd_t

    # 10. Time
    h_ = d["open_time"].dt.hour; dw = d["open_time"].dt.dayofweek
    F["sin_h"]   = np.sin(2*math.pi*h_/24); F["cos_h"]  = np.cos(2*math.pi*h_/24)
    F["sin_dow"] = np.sin(2*math.pi*dw/7);  F["cos_dow"]= np.cos(2*math.pi*dw/7)
    F["london"]  = h_.isin([8,9,10,11,12]).astype(float)
    F["ny"]      = h_.isin([13,14,15,16,17,18,19]).astype(float)
    F["weekend"] = (dw >= 4).astype(float)

    # 11. Funding
    avg_fr = 0.0; tr_fr = 0.0
    if fund is not None and len(fund) >= 3:
        rates  = fund["fundingRate"].tail(8).values.astype(float)
        avg_fr = float(rates.mean())
        tr_fr  = float(np.clip((rates[-1]-rates[0])*1000, -3, 3))
    F["fund_rate"]  = avg_fr; F["fund_trend"] = tr_fr
    F["fund_rev"]   = float(-1 if avg_fr>0.0008 else (1 if avg_fr<-0.0005 else 0))

    # 12. Stacked / liquidity
    F["stk_buy"]  = (dp>0.1).rolling(3).sum().eq(3).astype(float)
    F["stk_sell"] = (dp<-0.1).rolling(3).sum().eq(3).astype(float)
    F["bid_abs"]  = ((d["wick_bot"]>atr*0.25)&(dp>0.1)&(d["vol_z"]>1)).astype(float)
    F["ask_abs"]  = ((d["wick_top"]>atr*0.25)&(dp<-0.1)&(d["vol_z"]>1)).astype(float)

    # 13. Interactions
    F["mom_vol"]  = c_.pct_change(3) * d["vol_z"]
    F["dlt_mom"]  = dp * np.sign(c_.pct_change(1))
    F["vwap_dlt"] = F["vwap_dev_20"] * dp

    # 14. OU statistics
    ou_z = 0.0
    x_ou = c_.values[-100:] if len(c_) >= 100 else c_.values
    if len(x_ou) >= 30:
        dx_ = np.diff(x_ou); xl_ = x_ou[:-1]
        A_  = np.column_stack([np.ones(len(xl_)), xl_])
        try:
            co_,_,_,_ = np.linalg.lstsq(A_, dx_, rcond=None)
            mu_ou = -co_[0]/co_[1] if co_[1]!=0 else float(x_ou.mean())
            sg_ou = max(float(np.std(dx_-(co_[0]+co_[1]*xl_))), 1e-9)
            ou_z  = float(np.clip((float(x_ou[-1])-mu_ou)/sg_ou, -5, 5))
        except: pass
    F["ou_z"] = ou_z

    # Real-time tick features
    if tick_snap and tick_snap.get("trades", 0) > 5:
        bv  = tick_snap.get("buy_vol", 0)
        sv  = tick_snap.get("sell_vol", 0)
        F["tick_dp"]  = float(tick_snap.get("delta_pct", 0))
        F["tick_br"]  = float(bv / max(bv+sv+1e-9, 1))
        F["tick_pr"]  = float(1  if tick_snap.get("pressure","")=="BUY" else
                              -1 if tick_snap.get("pressure","")=="SELL" else 0)
    else:
        F["tick_dp"] = 0.0; F["tick_br"] = 0.5; F["tick_pr"] = 0.0

    return F.replace([np.inf, -np.inf], 0).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
#  TRIPLE-BARRIER + PURGED K-FOLD + CPCV SHARPE + GARCH + MARKET PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def triple_barrier(df, pct=0.008, t_max=5):
    prices = df["close"].astype(float).values
    atrs   = df["atr"].astype(float).values
    n      = len(prices); labels = np.full(n, np.nan)
    log_r  = np.diff(np.log(prices+1e-9))
    rv5    = np.array([log_r[max(0,i-5):i].std() if i>1 else pct for i in range(n)])
    rv5    = np.maximum(rv5, 0.002)
    for i in range(n - t_max):
        p0 = prices[i]; atr_i = atrs[i] if atrs[i]>0 else p0*0.003
        w  = max(pct, 1.5*rv5[i], atr_i/p0)
        tp = p0*(1+w); sl = p0*(1-w); lbl = 0
        for j in range(1, t_max+1):
            if i+j >= n: break
            p = prices[i+j]
            if p >= tp: lbl=1; break
            if p <= sl: lbl=-1; break
        if lbl == 0:
            rf = (prices[min(i+t_max, n-1)] / p0) - 1
            if   rf >  0.0005: lbl =  1
            elif rf < -0.0005: lbl = -1
        labels[i] = lbl
    return pd.Series(labels, index=df.index).dropna()

def purged_kfold(n, k=5, purge=5, embargo=2):
    fsize = n // k; splits = []
    for f in range(k):
        ts = f*fsize; te = ts+fsize if f<k-1 else n
        tr = list(range(0, max(0,ts-purge))) + list(range(min(n,te+embargo), n))
        ti = list(range(ts, te))
        if len(tr) >= 50 and len(ti) >= 10: splits.append((tr, ti))
    return splits

def cpcv_sharpe(oof_probs, y_dir, ret_s, n_splits=6, n_test=2):
    n = min(len(oof_probs), len(y_dir), len(ret_s))
    if n < 100: return 0.0
    fs = n // n_splits
    folds = [list(range(i*fs, (i+1)*fs if i<n_splits-1 else n)) for i in range(n_splits)]
    sharpes = []
    for combo in combinations(range(n_splits), n_test):
        ti = []
        for ci in combo: ti.extend(folds[ci])
        if len(ti) < 10: continue
        p_ = oof_probs[ti]; r_ = ret_s.iloc[ti].values
        st = np.where(p_>0.55, r_, np.where(p_<0.45, -r_, 0.0))
        sg = st.std(); mu = st.mean()
        sharpes.append(mu/sg*math.sqrt(288*252) if sg>0 else 0.0)
    return float(np.mean(sharpes)) if sharpes else 0.0

def garch11(ret):
    r = ret.dropna().values
    if len(r) < 30: return 0.003, 1.0, "MEDIUM", 50.0
    v0 = float(np.var(r))
    def nll(p):
        om, al, be = p
        if om<=0 or al<0 or be<0 or al+be>=1: return 1e10
        h = np.full(len(r), v0); ll = 0.0
        for t in range(1, len(r)):
            h[t] = om + al*r[t-1]**2 + be*h[t-1]
            if h[t] <= 0: return 1e10
            ll += -0.5*(math.log(2*math.pi*h[t]) + r[t]**2/h[t])
        return -ll
    try:
        res = optimize.minimize(nll, [v0*0.05,0.08,0.88], method="L-BFGS-B",
                                bounds=[(1e-9,None),(1e-9,0.999),(1e-9,0.999)],
                                options={"maxiter":100})
        om, al, be = res.x
    except: om, al, be = v0*0.05, 0.08, 0.88
    h = np.full(len(r), v0)
    for t in range(1, len(r)): h[t] = max(om+al*r[t-1]**2+be*h[t-1], 1e-12)
    cv  = float(math.sqrt(h[-1])); vp = float(stats.percentileofscore(np.sqrt(h), cv))
    rg  = "LOW" if vp<30 else ("HIGH" if vp>75 else "MEDIUM")
    sm  = 1.5 if vp<30 else (0.5 if vp>80 else 1.0)
    return cv, sm, rg, vp

def market_profile(df, tick=25.0):
    lo = df["low"].min(); hi = df["high"].max()
    bkts = np.arange(math.floor(lo/tick)*tick, math.ceil(hi/tick)*tick+tick, tick)
    vm = defaultdict(float)
    for _, row in df.iterrows():
        lvls = bkts[(bkts>=row["low"])&(bkts<=row["high"])]
        if not len(lvls): continue
        vp = row["volume"] / len(lvls)
        for lv in lvls: vm[lv] += vp
    if not vm: p = float(df["close"].iloc[-1]); return p, p, p
    pf  = pd.DataFrame({"p":list(vm.keys()),"v":list(vm.values())}).sort_values("p")
    poc = float(pf.loc[pf["v"].idxmax(),"p"]); tot = pf["v"].sum()
    pi  = pf["v"].idxmax(); cum = 0.0; va = []
    for _ in range(len(pf)):
        ui = pi+1; li = pi-1
        uv = pf.loc[ui,"v"] if ui in pf.index else 0.0
        dv = pf.loc[li,"v"] if li in pf.index else 0.0
        if uv>=dv and ui in pf.index: va.append(ui); cum+=uv; pi=ui
        elif li in pf.index: va.append(li); cum+=dv; pi=li
        else: break
        if cum/tot >= 0.70: break
    vah = float(pf.loc[va,"p"].max()) if va else poc+tick*5
    val = float(pf.loc[va,"p"].min()) if va else poc-tick*5
    return poc, vah, val

def kalman_filter(prices):
    z = prices.astype(float).values; n = len(z)
    F_ = np.array([[1.,1.],[0.,1.]]); H_ = np.array([[1.,0.]])
    Q_ = np.array([[0.01,0.001],[0.001,0.0001]]); R_ = np.array([[1.0]])
    x = np.array([[z[0]],[0.]]); P = np.eye(2)*1000.
    kp = np.zeros(n); kt = np.zeros(n)
    for t in range(n):
        xp = F_@x; Pp = F_@P@F_.T+Q_
        K  = Pp@H_.T@np.linalg.inv(H_@Pp@H_.T+R_)
        x  = xp + K*(z[t]-float((H_@xp).flat[0])); P=(np.eye(2)-K@H_)@Pp
        kp[t] = float(x[0].flat[0]); kt[t] = float(x[1].flat[0])
    return float(kp[-1]), float(kt[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  META-LABEL SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
class MetaLabelSystem:
    def __init__(self):
        self.primary    = None; self.secondary = None
        self.cal        = IsotonicRegression(out_of_bounds="clip")
        self.calibrated = False

    def fit(self, X, y_tb, splits, verbose=True):
        y_dir = (y_tb == 1).astype(int)
        gbm = GradientBoostingClassifier(n_estimators=CFG["GBM_N"], learning_rate=0.03,
              max_depth=4, subsample=0.70, min_samples_leaf=8, random_state=42)
        oof_p = np.full(len(X), 0.5)
        for tr, te in splits:
            y_tr = y_dir[tr]
            if len(np.unique(y_tr)) < 2: continue
            gbm.fit(X[tr], y_tr); oof_p[te] = gbm.predict_proba(X[te])[:, 1]
        if len(np.unique(y_dir)) < 2:
            yb = np.zeros(len(y_dir)); yb[len(y_dir)//2:] = 1; gbm.fit(X, yb)
        else:
            gbm.fit(X, y_dir)
        self.primary = gbm
        gbm_acc = float(((oof_p>0.5).astype(int)==y_dir).mean())
        if verbose: print("    GBM OOF: {:.4f}".format(gbm_acc))

        y_meta = np.zeros(len(y_tb)); pred_p = (oof_p>0.5).astype(int)
        for i in range(len(y_tb)):
            if   y_tb[i]==0:                          y_meta[i] = 0
            elif y_tb[i]==1  and pred_p[i]==1:        y_meta[i] = 1
            elif y_tb[i]==-1 and pred_p[i]==0:        y_meta[i] = 1
            else:                                      y_meta[i] = 0

        et = ExtraTreesClassifier(n_estimators=CFG["ET_N"], max_depth=5,
             min_samples_leaf=8, random_state=42, n_jobs=-1)
        oof_m = np.full(len(X), 0.5)
        for tr, te in splits:
            Xp_tr = np.column_stack([X[tr], oof_p[tr]])
            Xp_te = np.column_stack([X[te], oof_p[te]])
            ym    = y_meta[tr]
            if len(np.unique(ym)) < 2: continue
            et.fit(Xp_tr, ym); oof_m[te] = et.predict_proba(Xp_te)[:,1]
        valid = y_tb != 0
        if valid.sum() > 20:
            self.cal.fit(oof_m[valid], y_meta[valid]); self.calibrated = True
        et.fit(np.column_stack([X, oof_p]), y_meta); self.secondary = et
        ne      = y_tb != 0
        et_acc  = float(((oof_m[ne]>0.5).astype(int)==y_meta[ne]).mean()) if ne.sum()>0 else 0.5
        if verbose: print("    ET meta:  {:.4f}".format(et_acc))
        return oof_p, oof_m, gbm_acc, et_acc

    def predict(self, X, primary_prob):
        if not self.primary:
            return {"direction":0.5,"meta_prob":0.5,"signal":"WAIT","take":False}
        dir_p = float(self.primary.predict_proba(X[-1:])[:,1][0])
        Xp    = np.column_stack([X[-1:], [[primary_prob]]])
        meta  = float(self.secondary.predict_proba(Xp)[:,1][0]) if self.secondary else 0.5
        if self.calibrated: meta = float(self.cal.predict([meta])[0])
        return {"direction":dir_p,"meta_prob":meta,
                "signal":"BUY" if dir_p>0.55 else ("SELL" if dir_p<0.45 else "WAIT"),
                "take":meta >= CFG["MIN_META"]}


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADER + SIGNAL HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class PaperTrader:
    def __init__(self, account):
        self.balance    = account; self.start_bal = account
        self.position   = None;   self.trades    = []
        self.daily_pnl  = 0.0;    self.wins      = 0; self.losses = 0
        self.lock       = threading.Lock()

    @property
    def win_rate(self): return self.wins / max(self.wins+self.losses, 1) * 100
    @property
    def pnl_pct(self):  return (self.balance-self.start_bal)/self.start_bal*100

    def enter(self, side, entry, sl, tp1, tp2, qty, score, conf, reason):
        with self.lock:
            if self.position: return False
            slip = entry * CFG["PAPER_SLIP"] * (1 if side=="BUY" else -1)
            self.position = {"side":side,"entry":entry+slip,"sl":sl,"tp1":tp1,"tp2":tp2,
                             "qty":qty,"score":score,"conf":conf,"reason":reason,
                             "time":datetime.now(timezone.utc),"tp1_hit":False}
            return True

    def update(self, price):
        with self.lock:
            if not self.position: return None
            p = self.position; side = p["side"]; result = None
            if not p["tp1_hit"]:
                hit_tp1 = (side=="BUY" and price>=p["tp1"]) or (side=="SELL" and price<=p["tp1"])
                if hit_tp1:
                    pnl = p["qty"]*0.6*abs(p["tp1"]-p["entry"])*(1 if side=="BUY" else -1)
                    self.balance+=pnl; self.daily_pnl+=pnl
                    p["tp1_hit"]=True; p["qty"]*=0.4; p["sl"]=p["entry"]
                    result = {"type":"TP1","pnl":pnl,"price":price}
            if p["tp1_hit"]:
                hit_tp2 = (side=="BUY" and price>=p["tp2"]) or (side=="SELL" and price<=p["tp2"])
                if hit_tp2:
                    pnl = p["qty"]*abs(p["tp2"]-p["entry"])*(1 if side=="BUY" else -1)
                    self.balance+=pnl; self.daily_pnl+=pnl; self.wins+=1
                    self.trades.append({**p,"exit":price,"pnl":pnl,"result":"WIN"})
                    self.position=None
                    return {"type":"WIN","pnl":pnl,"price":price}
            hit_sl = (side=="BUY" and price<=p["sl"]) or (side=="SELL" and price>=p["sl"])
            if hit_sl:
                pnl = p["qty"]*abs(p["sl"]-p["entry"])*(-1 if side=="BUY" else 1)
                self.balance+=pnl; self.daily_pnl+=pnl; self.losses+=1
                self.trades.append({**p,"exit":price,"pnl":pnl,"result":"LOSS"})
                self.position=None
                result = {"type":"LOSS","pnl":pnl,"price":price}
            return result

    def stats(self):
        return {"balance":self.balance,"pnl_pct":self.pnl_pct,
                "trades":self.wins+self.losses,"wins":self.wins,"losses":self.losses,
                "win_rate":self.win_rate,"daily_pnl":self.daily_pnl,
                "in_position":self.position is not None}

class SignalHistory:
    def __init__(self, maxlen=300):
        self.signals = deque(maxlen=maxlen); self.correct=0; self.total=0
        self.lock    = threading.Lock()

    def record(self, side, price, score, conf, meta):
        with self.lock:
            self.signals.append({"side":side,"price":price,"score":score,"conf":conf,
                                  "meta_conf":meta,"time":datetime.now(timezone.utc),"outcome":None})

    def resolve(self, future_price):
        with self.lock:
            for sig in reversed(self.signals):
                if sig["outcome"] is None and sig["side"] != "WAIT":
                    ok = (sig["side"]=="BUY" and future_price>sig["price"]) or \
                         (sig["side"]=="SELL" and future_price<sig["price"])
                    sig["outcome"] = "WIN" if ok else "LOSS"
                    self.total += 1
                    if ok: self.correct += 1
                    break

    @property
    def live_acc(self): return self.correct/max(self.total,1)*100

    def recent(self, n=5):
        with self.lock: return list(self.signals)[-n:]


# ─────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET (tick + kline buffers)
# ─────────────────────────────────────────────────────────────────────────────
class TickBuffer:
    def __init__(self, maxlen=2000):
        self.ticks=deque(maxlen=maxlen); self.lock=threading.Lock()
        self.last_price=0.0; self.last_ts=0

    def add(self, price, qty, is_buyer_maker, ts):
        with self.lock:
            self.last_price=price; self.last_ts=ts
            self.ticks.append({"price":price,"qty":qty,"buy":not is_buyer_maker,"ts":ts})

    def snapshot(self, window_ms=30000):
        now = self.last_ts
        with self.lock:
            recent = [t for t in self.ticks if now-t["ts"] <= window_ms]
        if not recent:
            return {"buy_vol":0,"sell_vol":0,"delta":0,"delta_pct":0,"trades":0,
                    "price":self.last_price,"vwap":self.last_price,"pressure":"NEUTRAL"}
        bv   = sum(t["qty"] for t in recent if t["buy"])
        sv   = sum(t["qty"] for t in recent if not t["buy"])
        vwap = sum(t["price"]*t["qty"] for t in recent)/max(sum(t["qty"] for t in recent),1e-9)
        return {"buy_vol":bv,"sell_vol":sv,"delta":bv-sv,
                "delta_pct":float(np.clip((bv-sv)/(bv+sv+1e-9),-1,1)),
                "trades":len(recent),"price":self.last_price,"vwap":vwap,
                "pressure":"BUY" if bv>sv*1.3 else ("SELL" if sv>bv*1.3 else "NEUTRAL")}

class KlineBuffer:
    def __init__(self, maxlen=600):
        self.df=pd.DataFrame(); self.maxlen=maxlen; self.lock=threading.Lock()
        self.new_bar=threading.Event(); self.bar_count=0

    def update(self, row):
        with self.lock:
            nr = pd.DataFrame([row])
            nr["open_time"] = pd.to_datetime(nr["open_time"], unit="ms", utc=True)
            if self.df.empty: self.df=nr
            elif row["open_time"] not in self.df["open_time"].values:
                self.df=pd.concat([self.df,nr],ignore_index=True).tail(self.maxlen).reset_index(drop=True)
                self.bar_count+=1
            self.new_bar.set()

    def get_df(self):
        with self.lock: return self.df.copy()

    def wait(self, timeout=70):
        self.new_bar.clear(); return self.new_bar.wait(timeout=timeout)

class WSManager:
    def __init__(self, symbol, tf, kbuf, tbuf):
        self.sym=symbol.lower(); self.tf=tf; self.kbuf=kbuf; self.tbuf=tbuf
        self.connected=False; self._stop=threading.Event()
        self.ws_kl=None; self.ws_tick=None

    def _on_kline(self, ws, msg):
        try:
            data=json.loads(msg); k=data.get("k",{})
            if not k.get("x",False): return
            self.kbuf.update({"open_time":int(k["t"]),"open":float(k["o"]),
                               "high":float(k["h"]),"low":float(k["l"]),"close":float(k["c"]),
                               "volume":float(k["v"]),"taker_buy_vol":float(k.get("Q",float(k["v"])*0.5)),
                               "trades":int(k.get("n",0))})
        except Exception: pass

    def _on_tick(self, ws, msg):
        try:
            d=json.loads(msg)
            self.tbuf.add(float(d["p"]),float(d["q"]),bool(d["m"]),int(d["T"]))
        except Exception: pass

    def _run_kl(self):
        url="{}/{}@kline_{}".format(BASE_WS,self.sym,self.tf)
        while not self._stop.is_set():
            try:
                self.ws_kl=ws_module.WebSocketApp(url, on_message=self._on_kline,
                    on_open=lambda ws: setattr(self,"connected",True),
                    on_close=lambda ws,c,m: setattr(self,"connected",False),
                    on_error=lambda ws,e: setattr(self,"connected",False))
                self.ws_kl.run_forever(ping_interval=20,ping_timeout=10)
            except Exception: pass
            if not self._stop.is_set(): time.sleep(5)

    def _run_tick(self):
        url="{}/{}@aggTrade".format(BASE_WS,self.sym)
        while not self._stop.is_set():
            try:
                wst=ws_module.WebSocketApp(url,on_message=self._on_tick)
                wst.run_forever(ping_interval=20,ping_timeout=10)
                self.ws_tick=wst
            except Exception: pass
            if not self._stop.is_set(): time.sleep(5)

    def start(self):
        if not WS_OK: return False
        threading.Thread(target=self._run_kl,   daemon=True).start()
        threading.Thread(target=self._run_tick,  daemon=True).start()
        return True

    def stop(self):
        self._stop.set()
        if self.ws_kl:   self.ws_kl.close()
        if self.ws_tick: self.ws_tick.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(df, meta_res, resnet_p, gbm_prob, poc, vah, val,
              garch_m, vol_reg, tick_snap=None, math_r=None, bayes=None):
    price = float(df["close"].iloc[-1])
    atr   = float(df["atr"].iloc[-1]) or price*0.003
    ret   = df["close"].pct_change().dropna()
    dp    = df["delta_pct"].astype(float)
    dlt   = df["delta"].astype(float)

    # CVD divergence
    cvd20 = dlt.rolling(20).sum()
    pr_s  = df["close"].diff(3)/df["close"].shift(3)*100
    cvd_s = cvd20.diff(3)
    div_b = bool(pr_s.iloc[-1] < -0.12 and cvd_s.iloc[-1] > 0)
    div_s = bool(pr_s.iloc[-1] >  0.12 and cvd_s.iloc[-1] < 0)

    # OU z-score
    ou_z = 0.0
    x_ou = df["close"].values[-100:]
    if len(x_ou) >= 30:
        dx_ = np.diff(x_ou); xl_ = x_ou[:-1]
        A_  = np.column_stack([np.ones(len(xl_)), xl_])
        try:
            co_,_,_,_ = np.linalg.lstsq(A_, dx_, rcond=None)
            mu_ou = -co_[0]/co_[1] if co_[1]!=0 else float(x_ou.mean())
            sg_ou = max(float(np.std(dx_-(co_[0]+co_[1]*xl_))), 1e-9)
            ou_z  = float(np.clip((float(x_ou[-1])-mu_ou)/sg_ou, -5, 5))
        except: pass

    # Wyckoff
    n_w=min(30,len(df)); x_w=np.arange(n_w); rec=df.tail(n_w)
    def sl_(v):
        try: return float(np.polyfit(x_w[:len(v)],v,1)[0])
        except: return 0.0
    pt=sl_(rec["close"].values); bt=sl_(rec["taker_buy_vol"].values)
    st=sl_((rec["volume"]-rec["taker_buy_vol"]).values)
    wy=(3 if pt<-0.3 and bt>0 else 2 if pt>0.3 and bt>0 else
        -3 if pt>0.3 and st>0 else -2 if pt<-0.3 and st>0 else 0)

    kal_p, kal_t = kalman_filter(df["close"])

    bp   = df["body_pct"]; vz = float(df["vol_z"].iloc[-1])
    trap_s = bool(bp.shift(1).iloc[-1]<-0.25 and df["close"].iloc[-1]>df["open"].shift(1).iloc[-1])
    trap_l = bool(bp.shift(1).iloc[-1]> 0.25 and df["close"].iloc[-1]<df["open"].shift(1).iloc[-1])
    ab_sc  = (1 if vz>1.5 and dp.iloc[-1]>0.1  and abs(bp.iloc[-1])<0.08 else
             -1 if vz>1.5 and dp.iloc[-1]<-0.1 and abs(bp.iloc[-1])<0.08 else 0)

    # VWAP band
    c_   = df["close"].astype(float); vol_ = df["volume"].astype(float).replace(0,np.nan)
    tp__ = (df["high"]+df["low"]+c_)/3
    vw20 = (tp__*vol_).rolling(20).sum()/vol_.rolling(20).sum()
    vr20 = (vol_*(tp__-vw20)**2).rolling(20).sum()/vol_.rolling(20).sum()
    vs20 = np.sqrt(vr20.replace(0,np.nan))
    vdev = float((c_-vw20).iloc[-1]/vs20.iloc[-1]) if float(vs20.iloc[-1])>0 else 0.0
    vwap_sc = (2 if vdev<-1.8 else 1 if vdev<-0.8 else -2 if vdev>1.8 else -1 if vdev>0.8 else 0)

    # Tick
    tick_sc = 0
    if tick_snap and tick_snap.get("trades",0)>5:
        td = tick_snap.get("delta_pct",0)
        tick_sc = (2 if td>0.3 else 1 if td>0.1 else -2 if td<-0.3 else -1 if td<-0.1 else 0)

    # Individual scores
    resnet_sc = (3 if resnet_p>0.70 else 2 if resnet_p>0.62 else 1 if resnet_p>0.56 else
                -3 if resnet_p<0.30 else -2 if resnet_p<0.38 else -1 if resnet_p<0.44 else 0)
    dir_sc    = (3 if meta_res["direction"]>0.65 else 2 if meta_res["direction"]>0.56 else
                -3 if meta_res["direction"]<0.35 else -2 if meta_res["direction"]<0.44 else 0)
    meta_mult = (1.5 if meta_res["meta_prob"]>0.65 else 0.5 if meta_res["meta_prob"]<0.45 else 1.0)
    cvd_sc    = 3 if div_b else (-3 if div_s else 0)
    ou_sc     = (3 if ou_z<-2 else 2 if ou_z<-1 else 1 if ou_z<-0.5 else
                -3 if ou_z>2 else -2 if ou_z>1 else -1 if ou_z>0.5 else 0)
    kal_sc    = (2 if kal_t>0.2 else 1 if kal_t>0 else -2 if kal_t<-0.2 else -1 if kal_t<0 else 0)
    trap_sc   = (2 if trap_s else -2 if trap_l else 0)

    # Advanced math
    math_sc    = math_r.get("math_score",    0) if math_r else 0
    jump_pen   = math_r.get("jump_penalty",  1.0) if math_r else 1.0
    cvar_mult  = math_r.get("cvar_mult",     1.0) if math_r else 1.0

    raw = (resnet_sc*1.5 + dir_sc*meta_mult + cvd_sc + ou_sc + wy +
           kal_sc + trap_sc + ab_sc + vwap_sc + tick_sc + math_sc*0.5)
    raw *= (0.65 if vol_reg=="HIGH" else 1.0) * jump_pen
    score = int(np.clip(raw, -15, 15))
    conf  = min(abs(score)/15*100 * meta_res["meta_prob"]*1.8, 99.0)

    # Bayesian Kelly
    if bayes and MATH_OK:
        bay_win          = bayes.posterior_mean("resnet")
        bay_lo, bay_hi   = bayes.posterior_ci("resnet")
        kelly_adj        = AdvancedKelly.uncertainty_adjusted_kelly(bay_win, bay_lo, bay_hi, CFG["TP_MULT"])
    else:
        kelly_adj = 0.02
    kelly_final = float(np.clip(kelly_adj * garch_m * cvar_mult, 0, 0.08))

    stop_dist = atr * CFG["ATR_SL"]
    if score >= CFG["MIN_SCORE"]:
        side = "BUY"
        sl_  = round(min(val, price-stop_dist), 1)
        tp1  = round(poc if poc>price else price+stop_dist*CFG["TP_MULT"], 1)
        tp2  = round(vah if vah>tp1 else price+stop_dist*CFG["TP_MULT"]*2, 1)
    elif score <= -CFG["MIN_SCORE"]:
        side = "SELL"
        sl_  = round(max(vah, price+stop_dist), 1)
        tp1  = round(poc if poc<price else price-stop_dist*CFG["TP_MULT"], 1)
        tp2  = round(val if val<tp1 else price-stop_dist*CFG["TP_MULT"]*2, 1)
    else:
        side = "WAIT"; sl_=tp1=tp2=None

    rr   = abs(tp1-price)/max(abs(price-(sl_ or price)),1.0) if tp1 else 0.0
    qty  = (CFG["ACCOUNT"]*kelly_final/max(stop_dist,1.0)) if sl_ else 0.0
    ok   = (side!="WAIT" and conf>=CFG["MIN_CONF"] and rr>=CFG["MIN_RR"] and meta_res["take"])

    reasons = []
    if abs(resnet_sc)>=2: reasons.append("{:+.0f}ResNet".format(resnet_sc*1.5))
    if abs(dir_sc)>=2:    reasons.append("{:+.0f}Meta".format(dir_sc*meta_mult))
    if abs(cvd_sc)>=3:    reasons.append("{:+d}CVD".format(cvd_sc))
    if abs(ou_sc)>=2:     reasons.append("{:+d}OU".format(ou_sc))
    if abs(wy)>=2:        reasons.append("{:+d}Wyckoff".format(wy))
    if abs(math_sc)>=2:   reasons.append("{:+d}Math".format(math_sc))
    if abs(tick_sc)>=1:   reasons.append("{:+d}Tick".format(tick_sc))

    return {"side":side,"score":score,"confidence":conf,"tradeable":ok,
            "sl":sl_,"tp1":tp1,"tp2":tp2,"qty":round(qty,3),"rr":rr,
            "poc":poc,"vah":vah,"val":val,"garch_mult":garch_m,"vol_regime":vol_reg,
            "kelly":kelly_final,"ou_z":ou_z,"kal_trend":kal_t,"kal_price":kal_p,
            "meta_dir":meta_res["direction"],"meta_conf":meta_res["meta_prob"],
            "take_trade":meta_res["take"],"resnet_prob":resnet_p,"gbm_prob":gbm_prob,
            "div_bull":div_b,"div_bear":div_s,"wyckoff":wy,"trap":trap_l or trap_s,
            "vdev":vdev,"tick_sc":tick_sc,"math_sc":math_sc,
            "jump_pen":jump_pen,"cvar_mult":cvar_mult,
            "vol5":float(ret.tail(5).std()),"vol20":float(ret.tail(20).std()),
            "reasons":reasons,"tick_snap":tick_snap or {}}


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def display(price, res, tr, loop_n, live, cpcv_sh,
            paper_st, sig_hist, ws_conn, bayes_r,
            stoch_r, evt_r, info_r, online_acc,
            ckpt_info, drift):
    os.system("cls" if os.name=="nt" else "clear")
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    side = res["side"]; sc = res["score"]; conf = res["confidence"]
    sc_c = "G" if sc>0 else ("R" if sc<0 else "Y")
    ws_s = cc("WS:LIVE","G") if ws_conn else cc("REST","Y")
    ck_s = cc("SAVED","G")   if ckpt_info.get("saved") else cc("unsaved","D")
    dr_s = cc("DRIFT!","R")  if drift else cc("stable","D")

    print(cc("="*76, "C"))
    print(cc("  ULTIMATE QUANT ENGINE v5.0  |  BTC/USDT  |  Persistence + Real-Time", "C"))
    print(cc("  Bayes+EVT+InfoTheory+Particle+Copula+OnlineSGD+ResNet+TripleBarrier", "C"))
    print(cc("="*76, "C"))
    print("  {}  Bar#{}  {}  {}  {}  {}".format(
        cc(now,"D"), loop_n, "LIVE" if live else cc("SYNTH","Y"), ws_s, ck_s, dr_s))
    print("  {}  Vol5={:.3f}%  GARCH*{:.1f}  {}  Jump*{:.2f}  CVaR*{:.2f}".format(
        cc("${:,.2f}".format(price),"W"),
        res["vol5"]*100, res["garch_mult"],
        cc(res["vol_regime"],sc_c), res["jump_pen"], res["cvar_mult"]))
    print()

    # Tick
    ts = res.get("tick_snap",{}); n_trd = ts.get("trades",0)
    if n_trd > 0:
        tp = ts.get("pressure","NEUTRAL")
        tp_c = "G" if tp=="BUY" else ("R" if tp=="SELL" else "D")
        print(cc("  -- TICK FLOW (WebSocket aggTrade) ---------------------------------------","M"))
        print("  ${:,.2f}  VWAP${:,.2f}  Buy={:.2f}  Sell={:.2f}  d%={:+.3f}  {}  n={}".format(
            ts.get("price",price), ts.get("vwap",price),
            ts.get("buy_vol",0), ts.get("sell_vol",0),
            ts.get("delta_pct",0), cc(tp,tp_c), n_trd))
        print()

    # Main box
    b = bbar(abs(sc)/15)
    print(cc("  "+"="*68,"W"))
    if   side=="BUY":  print(cc("  ||  ####  B U Y  ^^^^^^^^^^^^  ####  Kelly={:.2f}%   ||".format(res["kelly"]*100),"G"))
    elif side=="SELL": print(cc("  ||  ####  S E L L  vvvvvvvvvvvv  ####  Kelly={:.2f}%  ||".format(res["kelly"]*100),"R"))
    else:              print(cc("  ||  ----  W A I T  (insufficient confluence)                ||","Y"))
    print("  ||  Score:{}  {}  Conf:{}  Meta:{}  Take:{}  ||".format(
        cc("{:>+3d}".format(sc),"B"), cc(b,sc_c),
        cc("{:.1f}%".format(conf),"B"),
        cc("{:.3f}".format(res["meta_conf"]),"B"),
        cc("YES","G") if res["take_trade"] else cc("NO","R")))
    print(cc("  "+"="*68,"W"))
    print()

    if res["tradeable"] and res["tp1"]:
        rr = res["rr"]; rrc = "G" if rr>=2.5 else ("Y" if rr>=1.5 else "R")
        print(cc("  +----- TRADE STRUCTURE -----------------------------------------------+","Y"))
        print("  |  Entry: ${:>12,.2f}{}|".format(price, " "*44))
        print(cc("  |  Stop:  ${:>12,.2f}  (${:>7,.1f} = {:.1f}x ATR)".format(
            res["sl"], abs(price-res["sl"]), CFG["ATR_SL"]),"R")+" "*19+cc("|","Y"))
        print(cc("  |  TP1:   ${:>12,.2f}  -> POC  (close 60%)".format(res["tp1"]),"G")+" "*20+cc("|","Y"))
        print(cc("  |  TP2:   ${:>12,.2f}  -> VAH/VAL (close 40%)".format(res["tp2"]),"G")+" "*17+cc("|","Y"))
        print("  |  R:R={}  Qty={:.3f}BTC  BayesKelly={:.2f}%  GARCH*{:.1f}{}|".format(
            cc("{:.2f}x".format(rr),rrc), res["qty"], res["kelly"]*100, res["garch_mult"], " "*8))
        print(cc("  +-------------------------------------------------------------------------+","Y"))
    elif side != "WAIT":
        print(cc("  Signal found but conf={:.1f}% / meta={:.3f} below threshold".format(conf,res["meta_conf"]),"Y"))
    print()

    # Advanced math
    print(cc("  -- ADVANCED MATH  (Bayes+Particle+EVT+InfoTheory+Copula) ----------------","M"))
    rows = [
        ("Bayesian P(win)",   "{:.4f}   BF={:.2f}  {}".format(
            bayes_r.get("posterior",0.5), bayes_r.get("bf",1),
            "STRONG EVIDENCE" if bayes_r.get("bf",1)>10 else ("evidence" if bayes_r.get("bf",1)>3 else "weak"))),
        ("Particle trend",    "{:>+.4f}/bar  {}".format(
            stoch_r.get("particle",{}).get("trend",0),
            stoch_r.get("particle",{}).get("trend_dir","?"))),
        ("RTS smoother",      "price={:,.1f}  trend={:>+.4f}".format(
            stoch_r.get("smoother",{}).get("current_price",price),
            stoch_r.get("smoother",{}).get("current_trend",0))),
        ("Heston long-vol",   "{:.4f}%  rho={:.3f}".format(
            stoch_r.get("heston",{}).get("long_run_vol",0)*100,
            stoch_r.get("heston",{}).get("rho",0))),
        ("Levy jumps",        "n={}  lambda={:.4f}  regime={}".format(
            stoch_r.get("levy",{}).get("n_jumps",0),
            stoch_r.get("levy",{}).get("lambda",0),
            "YES" if stoch_r.get("levy",{}).get("jump_regime") else "no")),
        ("EVT CVaR(99%)",     "{:.4f}%  tail={}  cvar_mult={:.2f}x".format(
            evt_r.get("cvar_99",-0.03)*100, evt_r.get("tail_regime","?"),
            evt_r.get("cvar_mult",1.0))),
        ("Info entropy",      "{:.3f}bits  regime={}  cvd_leads={}".format(
            info_r.get("entropy",0), info_r.get("market_regime","?"),
            "YES" if info_r.get("cvd_leads") else "no")),
        ("Online SGD acc",    "{:.2f}%  n_updates={}  drift={}".format(
            online_acc*100, stoch_r.get("n_updates",0),
            cc("YES!","R") if drift else "stable")),
        ("CPCV Sharpe",       "{:.3f}  {}".format(
            cpcv_sh, cc("STRONG EDGE","G") if cpcv_sh>1 else (
                     cc("WEAK EDGE","Y")   if cpcv_sh>0.3 else cc("NO EDGE","R")))),
    ]
    for lbl, val in rows:
        print("  {:<22} {}".format(lbl+":", val))
    print()

    # Bayesian posteriors
    if bayes_r.get("posteriors"):
        print(cc("  -- BAYESIAN POSTERIORS (updated every resolved trade) -------------------","M"))
        for nm, sd in list(bayes_r["posteriors"].items())[:6]:
            pm = sd.get("mean",0.5); lo = sd.get("ci_lo",0); hi = sd.get("ci_hi",1)
            bf = sd.get("bf",1); n = sd.get("n_obs",0)
            col = "G" if pm>0.55 else ("R" if pm<0.45 else "D")
            print("  {:<14} {}  CI=[{:.3f},{:.3f}]  BF={:.1f}  obs={}".format(
                nm, cc("P={:.3f}".format(pm),col), lo, hi, bf, n))
        print()

    # Active signals
    print(cc("  -- ACTIVE SIGNALS ---------------------------------------------------------","D"))
    def em(cond, text, col="G"):
        if cond: print("  {} {}".format(cc("*","Y"), cc(text,col)))

    em(res["resnet_prob"]>0.66, "RESNET BULL  P={:.4f}".format(res["resnet_prob"]))
    em(res["resnet_prob"]<0.34, "RESNET BEAR  P={:.4f}".format(res["resnet_prob"]), "R")
    em(res["div_bull"],         "CVD BULL DIVERGENCE  buyers accumulating")
    em(res["div_bear"],         "CVD BEAR DIVERGENCE  sellers distributing", "R")
    em(res["ou_z"]<-1.8,        "OU DOWN z={:.3f}  -> reversion BUY".format(res["ou_z"]))
    em(res["ou_z"]>1.8,         "OU UP   z={:.3f}  -> reversion SELL".format(res["ou_z"]), "R")
    em(res["wyckoff"]>=2,       "WYCKOFF {}".format("ACCUMULATION" if res["wyckoff"]==2 else "MARKUP"))
    em(res["wyckoff"]<=-2,      "WYCKOFF {}".format("DISTRIBUTION" if res["wyckoff"]==-2 else "MARKDOWN"), "R")
    em(res["trap"],             "TRAPPED TRADERS  squeeze incoming")
    em(res["kal_trend"]>0.2,    "KALMAN UP  {:.3f}/bar".format(res["kal_trend"]))
    em(res["kal_trend"]<-0.2,   "KALMAN DOWN {:.3f}/bar".format(res["kal_trend"]), "R")
    em(stoch_r.get("particle",{}).get("trend",0)>0.3,
       "PARTICLE FILTER BULL  trend={:.4f}".format(stoch_r.get("particle",{}).get("trend",0)))
    em(info_r.get("cvd_leads",False), "TRANSFER ENTROPY: CVD leads price  causal order flow")
    em(not res["take_trade"],   "META-LABEL REJECTS  skip this trade", "Y")
    em(drift,                   "CONCEPT DRIFT DETECTED  model updating", "Y")
    print()

    # Paper
    if paper_st:
        pc = "G" if paper_st["pnl_pct"]>=0 else "R"
        print(cc("  -- PAPER TRADING ----------------------------------------------------------","M"))
        print("  Balance:{}  PnL:{}  WR:{:.1f}%  Trades:{}  {}".format(
            cc("${:,.2f}".format(paper_st["balance"]),"W"),
            cc("{:+.2f}%".format(paper_st["pnl_pct"]),pc),
            paper_st["win_rate"], paper_st["trades"],
            cc("IN POSITION","G") if paper_st["in_position"] else ""))
        print()

    # Signal history
    recent = sig_hist.recent(5) if sig_hist else []
    if recent:
        print(cc("  -- SIGNAL HISTORY  acc:{:.1f}%  (n={}) ----------------------------------".format(
            sig_hist.live_acc, sig_hist.total),"D"))
        for s in reversed(recent):
            oc = s.get("outcome","—"); oc_c = "G" if oc=="WIN" else ("R" if oc=="LOSS" else "D")
            print("  {} {:>4}  {:+.0f}pts  conf={:.0f}%  meta={:.3f}  {}".format(
                s["time"].strftime("%H:%M:%S"), s["side"],
                s["score"], s["conf"], s["meta_conf"], cc(oc,oc_c)))
        print()

    print("  {} POC={:,.1f}  VAH={:,.1f}  VAL={:,.1f}  Kal={:,.1f}  CPCV={:.3f}".format(
        cc("o","C"), res["poc"], res["vah"], res["val"], res["kal_price"], cpcv_sh))
    print("  {} GBM={:.1f}%  ET={:.1f}%  ResNet={:.1f}%  PCA={}comp  n={}".format(
        cc("o","C"), tr.get("gbm_acc",0)*100, tr.get("et_acc",0)*100,
        tr.get("resnet_acc",0)*100, tr.get("n_pca",0), tr.get("n_samples",0)))
    print()
    print(cc("  -- REASONS ----------------------------------------------------------------","D"))
    print("  " + ("  |  ".join(res["reasons"]) if res["reasons"] else "Composite signal"))
    print()
    print(cc("  Ctrl+C  |  --paper  |  --account USDT  |  --tf 1m/5m  |  --reset","D"))
    print(cc("="*76,"D"))


# ─────────────────────────────────────────────────────────────────────────────
#  ULTIMATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class UltimateQuantEngine:

    def __init__(self, account=1000.0, paper=True, reset=False):
        CFG["ACCOUNT"] = account
        self.meta      = MetaLabelSystem()
        self.resnet    = None
        self.scaler    = RobustScaler()
        self.pca       = PCA(n_components=CFG["PCA_VAR"])
        self.trained   = False
        self.train_res = {}
        self.bar_count = 0
        self.bars_since_train  = 0
        self.bars_since_online = 0
        self.bars_since_ckpt   = 0
        self.cpcv_sh   = 0.0
        self.drift     = False
        self.X_last    = None

        # Advanced math
        if MATH_OK:
            self.math   = AdvancedMathSuite()
            self.bayes  = self.math.bayes
            self.online = self.math.online
        else:
            self.math   = None
            self.bayes  = BayesianEngine() if not MATH_OK else None
            self.online = None

        # Fallback simple Bayesian if uq_math not loaded
        if not MATH_OK:
            class SimpleBayes:
                def __init__(self):
                    self.posteriors = {"resnet":[2.0,2.0]}
                def update(self, sig, won):
                    if sig not in self.posteriors: self.posteriors[sig]=[2.0,2.0]
                    self.posteriors[sig][0 if won else 1] += 1
                def posterior_mean(self, sig):
                    a,b=self.posteriors.get(sig,[2.0,2.0]); return a/(a+b)
                def posterior_ci(self, sig):
                    a,b=self.posteriors.get(sig,[2.0,2.0])
                    from scipy.stats import beta as bd
                    return float(bd.ppf(0.05,a,b)), float(bd.ppf(0.95,a,b))
                def bayes_factor(self, sig): return 1.0
                def all_posteriors(self): return {}
            self.bayes = SimpleBayes()

        # Persistence
        self.store    = ModelStore(CFG["MODEL_DIR"])
        if reset: self.store.delete()

        # Buffers + WebSocket
        self.kbuf     = KlineBuffer(maxlen=600)
        self.tbuf     = TickBuffer(maxlen=2000)
        self.ws_mgr   = None
        self.ws_conn  = False

        # Paper + history
        self.paper    = PaperTrader(account) if paper else None
        self.sig_hist = SignalHistory(maxlen=300)
        self.ckpt_info= {"saved": False}

    # ── Full training ──────────────────────────────────────────────────────
    def train(self, df, fund, verbose=True):
        vp = verbose
        if vp:
            print(cc("\n  ULTIMATE QUANT ENGINE v5.0 — TRAINING","M"))
            print(cc("  "+"-"*60,"M"))

        if vp: print("  [1/6] Triple-barrier...", end=" ", flush=True)
        tb   = triple_barrier(df, pct=CFG["BARRIER_PCT"], t_max=CFG["TARGET_BARS"])
        idx  = tb.index; df_v = df.loc[idx]; y_tb = tb.values
        tp_r = float((y_tb==1).mean()*100); sl_r = float((y_tb==-1).mean()*100)
        ep_r = float((y_tb==0).mean()*100)
        if vp: print("TP={:.1f}%  SL={:.1f}%  Exp={:.1f}%".format(tp_r,sl_r,ep_r))

        if vp: print("  [2/6] Features...", end=" ", flush=True)
        F_df = build_features(df_v, fund, None)
        X_r  = np.nan_to_num(F_df.values.astype(float), 0.0)
        if vp: print("{} raw features".format(X_r.shape[1]))

        if vp: print("  [3/6] PCA...", end=" ", flush=True)
        X_sc  = self.scaler.fit_transform(X_r)
        X_pca = self.pca.fit_transform(X_sc)
        if vp: print("{} components ({:.0f}% var)".format(X_pca.shape[1], CFG["PCA_VAR"]*100))

        if vp: print("  [4/6] Purged K-Fold...")
        splits = purged_kfold(len(X_pca), k=5, purge=CFG["PURGE"], embargo=CFG["EMBARGO"])

        if vp: print("  [5/6] Meta-labeling...")
        oof_p, oof_m, gbm_acc, et_acc = self.meta.fit(X_pca, y_tb, splits, verbose=vp)

        if vp: print("  [6/6] ResNet...", end=" ", flush=True)
        mask = y_tb != 0; X_nn = X_pca[mask]; y_nn = (y_tb[mask]==1).astype(float)
        resnet_acc = 0.5
        if len(X_nn) > 80:
            nv = max(int(len(X_nn)*0.15), 20)
            self.resnet = ResNet(n_in=X_nn.shape[1], hidden=CFG["NN_H"],
                                  n_blocks=CFG["NN_B"], lr=CFG["NN_LR"],
                                  l2=CFG["NN_L2"], dropout=CFG["NN_DR"])
            self.resnet.fit(X_nn[:-nv], y_nn[:-nv],
                            Xv=X_nn[-nv:], yv=y_nn[-nv:], epochs=CFG["NN_EP"])
            resnet_acc = self.resnet.val_acc
        if vp: print("val_acc={:.4f}".format(resnet_acc))

        # Online initial fit
        if self.online:
            y_dir = (y_tb==1).astype(int)
            mask_ne = y_tb != 0
            if mask_ne.sum() > 50 and len(np.unique(y_dir[mask_ne])) >= 2:
                self.online.partial_fit(X_pca[mask_ne], y_dir[mask_ne])

        if vp: print("  CPCV Sharpe...", end=" ", flush=True)
        y_dir = (y_tb==1).astype(int)
        ret_s = df["close"].pct_change().fillna(0); ret_loc = ret_s.loc[idx]
        self.cpcv_sh = (cpcv_sharpe(oof_p, y_dir, ret_loc)
                        if len(np.unique(y_dir)) >= 2 else 0.0)
        if vp: print("{:.3f}".format(self.cpcv_sh))

        self.trained  = True
        self.X_last   = X_pca
        self.train_res= {"n_raw":X_r.shape[1],"n_pca":X_pca.shape[1],
                          "n_samples":len(X_pca),"gbm_acc":gbm_acc,
                          "et_acc":et_acc,"resnet_acc":resnet_acc,
                          "tb_tp":tp_r,"tb_sl":sl_r,"tb_exp":ep_r}
        if vp: print(cc("\n  Done.  CPCV Sharpe={:.3f}\n".format(self.cpcv_sh),"G"))
        self._checkpoint(self.cpcv_sh)

    # ── Online update ──────────────────────────────────────────────────────
    def online_update(self, df, fund):
        if not self.online or not self.trained: return
        try:
            F_df = build_features(df.tail(40), fund, None)
            X_r  = np.nan_to_num(F_df.values.astype(float), 0.0)
            if X_r.shape[0] < 10: return
            X_sc  = self.scaler.transform(X_r)
            X_pca = self.pca.transform(X_sc)
            tb    = triple_barrier(df.tail(40), pct=CFG["BARRIER_PCT"], t_max=CFG["TARGET_BARS"])
            y_tb  = tb.values; y_dir = (y_tb==1).astype(int); mask = y_tb != 0
            if mask.sum() < 5 or len(np.unique(y_dir[mask])) < 2: return
            res = self.online.partial_fit(X_pca[mask], y_dir[mask])
            self.drift = res.get("drift", False)
            if self.drift:
                print(cc("\n  [DRIFT] Concept drift detected — model adapting","Y"))
        except Exception: pass

    # ── Checkpoint ────────────────────────────────────────────────────────
    def _checkpoint(self, sharpe=None):
        state = {
            "meta_primary":    self.meta.primary,
            "meta_secondary":  self.meta.secondary,
            "meta_calibrated": self.meta.calibrated,
            "meta_cal":        self.meta.cal,
            "scaler":          self.scaler,
            "pca":             self.pca,
            "resnet_weights":  self.resnet.get_weights() if self.resnet else {},
            "resnet_config":   {"n_in": self.X_last.shape[1] if self.X_last is not None else 0,
                                 "hidden": CFG["NN_H"], "n_blocks": CFG["NN_B"]},
            "bayes_posteriors":self.bayes.posteriors,
            "online_sgd":      (self.online.sgd if self.online and self.online.initialized else None),
            "online_scaler":   (self.online.scaler if self.online else None),
            "cpcv_sharpe":     self.cpcv_sh,
            "train_res":       self.train_res,
            "n_samples":       self.train_res.get("n_samples", 0),
        }
        saved = self.store.save(state, sharpe, tag="latest")
        self.ckpt_info = {"saved": saved, "sharpe": sharpe or 0,
                           "time": datetime.now(timezone.utc).strftime("%H:%M:%S")}

    # ── Load checkpoint ────────────────────────────────────────────────────
    def _load(self):
        state = self.store.load(prefer_best=True)
        if not state: return False
        try:
            self.meta.primary     = state.get("meta_primary")
            self.meta.secondary   = state.get("meta_secondary")
            self.meta.calibrated  = state.get("meta_calibrated", False)
            if "meta_cal" in state: self.meta.cal = state["meta_cal"]
            self.scaler           = state.get("scaler", RobustScaler())
            self.pca              = state.get("pca", PCA(n_components=CFG["PCA_VAR"]))
            if hasattr(self.bayes, "posteriors"):
                self.bayes.posteriors = state.get("bayes_posteriors", self.bayes.posteriors)
            self.cpcv_sh          = state.get("cpcv_sharpe", 0.0)
            self.train_res        = state.get("train_res", {})

            rw = state.get("resnet_weights", {}); rc = state.get("resnet_config", {})
            n_in = rc.get("n_in", 0)
            if rw and n_in > 0:
                self.resnet = ResNet(n_in=n_in, hidden=rc.get("hidden",64),
                                      n_blocks=rc.get("n_blocks",3))
                self.resnet.set_weights(rw)

            if self.online and state.get("online_sgd") is not None:
                self.online.sgd         = state["online_sgd"]
                self.online.initialized = True
                if state.get("online_scaler"): self.online.scaler = state["online_scaler"]

            self.trained = self.meta.primary is not None
            print(cc("  [LOAD] Model restored. CPCV={:.3f}  Trained={}".format(
                self.cpcv_sh, self.trained), "G"))
            return True
        except Exception as e:
            print("  [LOAD ERROR] {}".format(e))
            return False

    # ── Inference ─────────────────────────────────────────────────────────
    def infer(self, df, fund, tick_snap=None):
        F_df   = build_features(df, fund, tick_snap)
        X_raw  = np.nan_to_num(F_df.values.astype(float), 0.0)
        X_sc   = self.scaler.transform(X_raw)
        X_pca  = self.pca.transform(X_sc)

        gbm_p    = float(self.meta.primary.predict_proba(X_pca[-1:])[:,1][0]) if self.meta.primary else 0.5
        meta_res = self.meta.predict(X_pca, gbm_p)
        rnet_p   = float(self.resnet.predict(X_pca[-1:])[0]) if self.resnet else 0.5

        # Advanced math
        math_r = {}; stoch_r = {}; evt_r = {}; info_r = {}
        if self.math:
            try:
                math_r  = self.math.run_all(df, X_pca)
                stoch_r = math_r.get("stoch", {})
                evt_r   = math_r.get("evt",   {})
                info_r  = math_r.get("info",  {})
            except Exception as e:
                pass

        poc, vah, val = market_profile(df)
        ret = df["close"].pct_change().dropna()
        _, garch_m, vol_reg, _ = garch11(ret)

        res = aggregate(df, meta_res, rnet_p, gbm_p, poc, vah, val,
                        garch_m, vol_reg, tick_snap, math_r, self.bayes)
        return res, stoch_r, evt_r, info_r

    # ── Main run loop ─────────────────────────────────────────────────────
    def run(self):
        live = False; fund = pd.DataFrame(); df = pd.DataFrame()

        # WebSocket
        if WS_OK and NET:
            print(cc("  Starting WebSocket streams...","M"), flush=True)
            self.ws_mgr = WSManager(CFG["SYMBOL"], CFG["TF"], self.kbuf, self.tbuf)
            self.ws_mgr.start(); time.sleep(3)
            self.ws_conn = self.ws_mgr.connected
            print("  WS connected: {}".format(self.ws_conn))

        # REST bootstrap
        if NET:
            try:
                df   = fetch_klines(CFG["SYMBOL"], CFG["TF"], CFG["CANDLES"])
                fund = fetch_funding(CFG["SYMBOL"]); live = True
                print("  REST data: {} bars".format(len(df)))
            except Exception as e:
                print("  REST error: {}  -> synthetic".format(e))

        if df.empty: df, fund = make_synthetic(seed=42)
        df = prepare(df)

        # Load or train
        if self.store.exists():
            print(cc("  Checkpoint found — loading...","Y"))
            if not self._load():
                print(cc("  Load failed — training fresh","Y"))
                self.train(df, fund, verbose=True)
        else:
            print(cc("  No checkpoint — training fresh","Y"))
            self.train(df, fund, verbose=True)

        # Seed kline buffer from REST data
        for _, row in df.tail(200).iterrows():
            self.kbuf.update({"open_time": int(row["open_time"].timestamp()*1000),
                               "open":float(row["open"]), "high":float(row["high"]),
                               "low":float(row["low"]),   "close":float(row["close"]),
                               "volume":float(row["volume"]),
                               "taker_buy_vol":float(row["taker_buy_vol"]),
                               "trades":int(row["trades"])})

        print(cc("\n  Real-time inference loop started...\n","G"))
        curr_df = df

        while True:
            try:
                # Wait for bar
                if self.ws_mgr and self.ws_conn:
                    self.ws_conn = self.ws_mgr.connected
                    if not self.kbuf.wait(timeout=70): continue
                    curr_df = self.kbuf.get_df()
                    if curr_df.empty or len(curr_df) < 50: continue
                    curr_df = prepare(curr_df)
                else:
                    time.sleep(30)
                    if NET:
                        try:
                            df   = fetch_klines(CFG["SYMBOL"],CFG["TF"],CFG["CANDLES"])
                            fund = fetch_funding(CFG["SYMBOL"]); live=True
                        except: pass
                    curr_df = prepare(df)

                self.bar_count         += 1
                self.bars_since_train  += 1
                self.bars_since_online += 1
                self.bars_since_ckpt   += 1
                price = float(curr_df["close"].iloc[-1])

                # Retrain check
                if self.bars_since_train >= CFG["RETRAIN_N"]:
                    print(cc("\n  [RETRAIN] After {} bars...".format(self.bars_since_train),"Y"))
                    self.train(curr_df, fund, verbose=False)
                    self.bars_since_train = 0

                # Online update
                elif self.bars_since_online >= CFG["ONLINE_N"]:
                    self.online_update(curr_df, fund)
                    self.bars_since_online = 0

                # Periodic checkpoint
                if self.bars_since_ckpt >= CFG["CHECKPOINT_N"]:
                    self._checkpoint(self.cpcv_sh)
                    self.bars_since_ckpt = 0

                # Tick snapshot
                tick_snap = self.tbuf.snapshot(CFG["TICK_WIN_MS"])

                # Infer
                res, stoch_r, evt_r, info_r = self.infer(curr_df, fund, tick_snap)

                # Resolve signal history
                if self.bar_count > 5: self.sig_hist.resolve(price)

                # Paper trading
                if self.paper:
                    tr_result = self.paper.update(price)
                    if tr_result:
                        won = tr_result["type"] in ["WIN","TP1"]
                        for sn in ["resnet","gbm","cvd_div","ou_rev","wyckoff","kalman"]:
                            self.bayes.update(sn, won)
                        pnl_str = "${:+.2f}".format(tr_result["pnl"])
                        print(cc("  [PAPER] {}  {}  @${:,.2f}".format(
                            tr_result["type"], pnl_str, tr_result["price"]),
                            "G" if won else "R"))
                    if res["tradeable"] and not self.paper.position:
                        entered = self.paper.enter(
                            res["side"], price, res["sl"], res["tp1"], res["tp2"],
                            res["qty"], res["score"], res["confidence"],
                            " | ".join(res["reasons"]))
                        if entered:
                            self.sig_hist.record(res["side"], price, res["score"],
                                                  res["confidence"], res["meta_conf"])
                            print(cc("  [PAPER] Entered {} @${:,.2f}  sl={:,.1f}  tp1={:,.1f}".format(
                                res["side"], price, res["sl"] or 0, res["tp1"] or 0),"G"))

                # Bayesian summary
                bayes_r = {
                    "posterior":  self.bayes.posterior_mean("resnet"),
                    "bf":         self.bayes.bayes_factor("resnet"),
                    "posteriors": self.bayes.all_posteriors(),
                }

                # Display
                paper_st   = self.paper.stats() if self.paper else None
                online_acc = self.online.rolling_accuracy() if self.online else 0.5
                display(price, res, self.train_res, self.bar_count, live, self.cpcv_sh,
                        paper_st, self.sig_hist, self.ws_conn, bayes_r,
                        stoch_r, evt_r, info_r, online_acc, self.ckpt_info, self.drift)

            except KeyboardInterrupt:
                print(cc("\n  Stopped by user.","Y"))
                if self.ws_mgr: self.ws_mgr.stop()
                self._checkpoint(self.cpcv_sh)
                print(cc("  Final checkpoint saved.","G"))
                if self.paper:
                    st = self.paper.stats()
                    print("  Final: ${:,.2f}  PnL={:+.2f}%  WR={:.1f}%  Trades={}".format(
                        st["balance"], st["pnl_pct"], st["win_rate"], st["trades"]))
                break
            except Exception as exc:
                import traceback; print("  Error: {}".format(exc))
                traceback.print_exc(); time.sleep(15)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Ultimate Quant Engine v5.0")
    p.add_argument("--account",    type=float, default=1000.0, help="Account size USDT")
    p.add_argument("--paper",      action="store_true",        help="Paper trading mode")
    p.add_argument("--tf",         type=str,   default="5m",   help="Timeframe: 1m 3m 5m 15m")
    p.add_argument("--symbol",     type=str,   default="BTCUSDT")
    p.add_argument("--reset",      action="store_true",        help="Delete saved model")
    p.add_argument("--retrain",    type=int,   default=100,    help="Full retrain every N bars")
    p.add_argument("--checkpoint", type=int,   default=30,     help="Save checkpoint every N bars")
    p.add_argument("--online",     type=int,   default=5,      help="Online update every N bars")
    p.add_argument("--model-dir",  type=str,   default="uq_models", dest="model_dir")
    a = p.parse_args()

    CFG["TF"]           = a.tf
    CFG["SYMBOL"]       = a.symbol
    CFG["ACCOUNT"]      = a.account
    CFG["RETRAIN_N"]    = a.retrain
    CFG["CHECKPOINT_N"] = a.checkpoint
    CFG["ONLINE_N"]     = a.online
    CFG["MODEL_DIR"]    = a.model_dir

    print(cc("\n" + "="*76, "C"))
    print(cc("  ULTIMATE QUANT ENGINE v5.0", "C"))
    print(cc("  Bayes+EVT+InfoTheory+Particle+Copula+Heston+OnlineSGD+ResNet+CPCV", "C"))
    print(cc("  Model Persistence + Real-Time WebSocket + Paper Trading", "C"))
    print(cc("="*76, "C"))
    print("  Symbol:     {}".format(CFG["SYMBOL"]))
    print("  Timeframe:  {}".format(CFG["TF"]))
    print("  Account:    ${:,.2f} USDT".format(a.account))
    print("  Mode:       {}".format("PAPER TRADING" if a.paper else "SIGNALS ONLY"))
    print("  Model dir:  {}".format(CFG["MODEL_DIR"]))
    print("  WebSocket:  {}".format("available" if WS_OK else "NOT available (pip install websocket-client)"))
    print("  Math suite: {}".format("LOADED" if MATH_OK else "NOT loaded (uq_math.py missing)"))
    print()

    engine = UltimateQuantEngine(account=a.account, paper=a.paper, reset=a.reset)
    engine.run()


if __name__ == "__main__":
    main()
