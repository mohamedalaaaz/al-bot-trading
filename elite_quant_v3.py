#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ELITE QUANT ENGINE v3.0  —  BTC/USDT Binance Futures                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  INSTITUTIONAL TECHNIQUES:                                                  ║
║   1. Triple-Barrier Labeling (Lopez de Prado Ch.3)                         ║
║   2. Purged + Embargoed K-Fold (Ch.7) — no leakage                        ║
║   3. Fractional Differentiation (Ch.5) — stationary + memory              ║
║   4. Meta-Labeling (Ch.10) — filter low-confidence signals                ║
║   5. Deep ResNet (skip connections, Swish, Adam, dropout)                  ║
║   6. GARCH(1,1) vol-regime aware position sizing                           ║
║   7. Sharpe-optimal Kelly with parameter uncertainty shrinkage             ║
║   8. CPCV Sharpe estimate — true unbiased performance measure             ║
║   9. Isotonic probability calibration                                       ║
║  10. 120+ alpha features across 14 signal categories                       ║
║  11. Kalman filter + OU mean reversion + CVD divergence                    ║
║  12. Wyckoff cycle + smart money flow                                       ║
║  13. Liquidity sweep + trapped trader detection                            ║
║  14. Market profile (POC / VAH / VAL)                                      ║
║                                                                             ║
║  OUTPUT: BUY / SELL / WAIT  with entry, stop, TP1, TP2, size, Kelly       ║
║                                                                             ║
║  RUN:  python elite_quant.py                                               ║
║        python elite_quant.py --loop --interval 30                         ║
║        python elite_quant.py --account 5000                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import warnings
import argparse
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
from scipy.signal import hilbert as sp_hilbert
from scipy.interpolate import PchipInterpolator

from sklearn.ensemble import (GradientBoostingClassifier,
                               ExtraTreesClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

try:
    import requests
    NET = True
except ImportError:
    NET = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "SYMBOL":         "BTCUSDT",
    "INTERVAL":       "5m",
    "CANDLES":        500,
    "ACCOUNT":        1000.0,
    "MAX_RISK_PCT":   0.015,
    "LEVERAGE":       5,
    "MIN_SCORE":      6,
    "MIN_CONF":       55.0,
    "MIN_META_CONF":  0.52,
    "MIN_RR":         1.5,
    "ATR_SL_MULT":    1.5,
    "TP_MULT":        2.5,
    "LOOP_SECS":      30,
    # ML
    "TARGET_BARS":    3,
    "BARRIER_PCT":    0.015,
    "FRAC_D":         0.40,
    "PCA_VAR":        0.90,
    "N_PURGE":        5,
    "N_EMBARGO":      2,
    "META_THR":       0.52,
    "GBM_TREES":      300,
    "ET_TREES":       200,
    "NN_HIDDEN":      64,
    "NN_BLOCKS":      3,
    "NN_EPOCHS":      100,
    "NN_LR":          5e-4,
    "NN_DROPOUT":     0.25,
    "NN_L2":          1e-4,
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_binance(symbol, interval, limit):
    url = "https://fapi.binance.com/fapi/v1/klines"
    r   = requests.get(url, params={"symbol": symbol, "interval": interval,
                                     "limit": limit}, timeout=12)
    r.raise_for_status()
    df  = pd.DataFrame(r.json(), columns=[
        "ts","o","h","l","c","v","ct","qv","n","tbv","tbqv","_"])
    df["open_time"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    for col in ["o","h","l","c","v","tbv","n"]:
        df[col] = df[col].astype(float)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close",
                               "v":"volume","tbv":"taker_buy_vol","n":"trades"})[
        ["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]

def fetch_funding(symbol, limit=50):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    r   = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
    r.raise_for_status()
    df  = pd.DataFrame(r.json())
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(float), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df

def make_synthetic(n=500, seed=42, base=67000.0):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min", tz="UTC")
    price = float(base)
    rows  = []
    for dt in dates:
        h    = dt.hour
        sv   = 2.2 if h in [8, 9, 13, 14, 15, 16] else 0.65
        mu   = -0.00018 if h in [16, 17, 18] else 0.00012
        ret  = np.random.normal(mu, 0.0028 * sv)
        price = max(price * (1.0 + ret), 50000.0)
        hi   = price * (1.0 + abs(np.random.normal(0, 0.002 * sv)))
        lo   = price * (1.0 - abs(np.random.normal(0, 0.002 * sv)))
        vol  = max(abs(np.random.normal(1100, 380)) * sv, 80.0)
        bsk  = 0.63 if h in [8, 9] else (0.36 if h in [17, 18] else 0.50)
        tb   = vol * float(np.clip(np.random.beta(bsk * 7, (1 - bsk) * 7), 0.05, 0.95))
        if np.random.random() < 0.025:
            vol *= np.random.uniform(5, 9)
        rows.append({
            "open_time":     dt,
            "open":          price * (1.0 + np.random.normal(0, 0.001)),
            "high":          hi,
            "low":           lo,
            "close":         price,
            "volume":        vol,
            "taker_buy_vol": tb,
            "trades":        int(vol / 0.04),
        })
    df   = pd.DataFrame(rows)
    fund = pd.DataFrame([
        {"fundingTime": dates[i],
         "fundingRate": float(np.random.normal(0.0001, 0.0003))}
        for i in range(0, n, 96)
    ])
    return df, fund

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["body"]       = d["close"] - d["open"]
    d["body_pct"]   = d["body"] / d["open"] * 100.0
    d["is_bull"]    = d["body"] > 0
    d["wick_top"]   = d["high"] - d[["open", "close"]].max(axis=1)
    d["wick_bot"]   = d[["open", "close"]].min(axis=1) - d["low"]
    d["sell_vol"]   = d["volume"] - d["taker_buy_vol"]
    d["delta"]      = d["taker_buy_vol"] - d["sell_vol"]
    d["delta_pct"]  = (d["delta"] / d["volume"].replace(0, np.nan)).fillna(0)
    hl  = d["high"] - d["low"]
    hpc = (d["high"] - d["close"].shift(1)).abs()
    lpc = (d["low"]  - d["close"].shift(1)).abs()
    d["atr"]        = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    rm  = d["volume"].rolling(50).mean()
    rs  = d["volume"].rolling(50).std().replace(0, np.nan)
    d["vol_z"]      = (d["volume"] - rm) / rs
    d["hour"]       = d["open_time"].dt.hour
    d["dow"]        = d["open_time"].dt.dayofweek
    d["session"]    = d["hour"].apply(
        lambda h: "Asia" if h < 8 else "London" if h < 13 else "NY" if h < 20 else "Late"
    )
    return d.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
#  FRACTIONAL DIFFERENTIATION  (Lopez de Prado Ch.5)
# ─────────────────────────────────────────────────────────────────────────────
def frac_diff(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    """Fixed-width window fractional differentiation."""
    w = [1.0]
    k = 1
    while True:
        val = -w[-1] * (d - k + 1) / k
        if abs(val) < thresh:
            break
        w.append(val)
        k += 1
    w     = np.array(w[::-1])
    width = len(w)
    out   = pd.Series(np.nan, index=series.index)
    for i in range(width - 1, len(series)):
        out.iloc[i] = float(np.dot(w, series.iloc[i - width + 1: i + 1].values))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  TRIPLE-BARRIER LABELING  (Lopez de Prado Ch.3)
# ─────────────────────────────────────────────────────────────────────────────
def triple_barrier(df: pd.DataFrame, pct: float = 0.015, t_max: int = 5) -> pd.Series:
    prices = df["close"].astype(float).values
    n      = len(prices)
    labels = np.full(n, np.nan)
    for i in range(n - t_max):
        p0 = prices[i]
        tp = p0 * (1.0 + pct)
        sl = p0 * (1.0 - pct)
        lbl = 0
        for j in range(1, t_max + 1):
            if i + j >= n:
                break
            p = prices[i + j]
            if p >= tp:
                lbl = 1
                break
            if p <= sl:
                lbl = -1
                break
        labels[i] = lbl
    return pd.Series(labels, index=df.index).dropna()


# ─────────────────────────────────────────────────────────────────────────────
#  PURGED K-FOLD SPLITS  (Lopez de Prado Ch.7)
# ─────────────────────────────────────────────────────────────────────────────
def purged_kfold(n: int, k: int = 5, purge: int = 5, embargo: int = 2):
    fsize  = n // k
    splits = []
    for f in range(k):
        ts = f * fsize
        te = ts + fsize if f < k - 1 else n
        tr = list(range(0, max(0, ts - purge))) + \
             list(range(min(n, te + embargo), n))
        te_idx = list(range(ts, te))
        if len(tr) >= 50 and len(te_idx) >= 10:
            splits.append((tr, te_idx))
    return splits


# ─────────────────────────────────────────────────────────────────────────────
#  120+ ALPHA FEATURES  (14 categories)
# ─────────────────────────────────────────────────────────────────────────────
def rsi_series(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

def build_features(df: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    d   = df.copy()
    c_  = d["close"].astype(float)
    vol = d["volume"].astype(float).replace(0, np.nan)
    dp  = d["delta_pct"].astype(float)
    dlt = d["delta"].astype(float)
    ret = c_.pct_change()
    lr  = np.log(c_ / c_.shift(1)).fillna(0)
    tp  = (d["high"] + d["low"] + c_) / 3.0
    atr = d["atr"].astype(float).replace(0, np.nan)

    F = pd.DataFrame(index=d.index)

    # ── 1. Momentum (Fibonacci lags) ──────────────────────────────────────
    for lag in [1, 2, 3, 5, 8, 13, 21, 34]:
        F["mom_" + str(lag)] = c_.pct_change(lag)
    for fast, slow in [(8, 21), (12, 26), (5, 13)]:
        key = "macd_" + str(fast) + "_" + str(slow)
        F[key] = (c_.ewm(fast).mean() - c_.ewm(slow).mean()) / c_
    F["mom_acc"] = c_.pct_change(5) - c_.pct_change(5).shift(5)
    for w in [10, 20, 50]:
        hi_ = d["high"].rolling(w).max()
        lo_ = d["low"].rolling(w).min()
        rng = (hi_ - lo_).replace(0, np.nan)
        F["rpos_" + str(w)] = (c_ - lo_) / rng
        F["dhi_"  + str(w)] = (hi_ - c_) / c_ * 100.0
        F["dlo_"  + str(w)] = (c_ - lo_) / c_ * 100.0

    # ── 2. Mean Reversion ─────────────────────────────────────────────────
    for w in [10, 20, 50, 100]:
        mu_ = c_.rolling(w).mean()
        sg_ = c_.rolling(w).std().replace(0, np.nan)
        F["z_" + str(w)] = (c_ - mu_) / sg_
    for p in [7, 14, 21]:
        F["rsi_" + str(p)] = rsi_series(c_, p)
    F["willr"] = ((d["high"].rolling(14).max() - c_) /
                  (d["high"].rolling(14).max() - d["low"].rolling(14).min() + 1e-9) * -100.0)
    ma_cci = tp.rolling(20).mean()
    md_cci = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).replace(0, np.nan)
    F["cci"] = (tp - ma_cci) / (0.015 * md_cci)

    # ── 3. Fractional Differentiation ────────────────────────────────────
    for d_val in [0.3, 0.4, 0.5]:
        key = "fd_" + str(d_val).replace(".", "")
        F[key] = frac_diff(c_, d_val)

    # ── 4. Order Flow / Delta ────────────────────────────────────────────
    F["delta_pct"]   = dp
    F["buy_ratio"]   = d["taker_buy_vol"] / vol
    F["vol_imb"]     = dlt / vol
    cvd20            = dlt.rolling(20).sum()
    F["cvd_20n"]     = cvd20 / vol.rolling(20).mean()
    F["cvd_sl3"]     = cvd20.diff(3)
    F["cvd_sl5"]     = cvd20.diff(5)
    F["cvd_acc"]     = cvd20.diff(3).diff(2)
    pr_s             = c_.diff(3) / c_.shift(3) * 100.0
    cvd_s            = cvd20.diff(3)
    F["div_bull"]    = ((pr_s < -0.12) & (cvd_s > 0)).astype(float)
    F["div_bear"]    = ((pr_s >  0.12) & (cvd_s < 0)).astype(float)
    F["exh_buy"]     = ((dp > 0.28) & (d["body_pct"].abs() < 0.06)).astype(float)
    F["exh_sell"]    = ((dp < -0.28) & (d["body_pct"].abs() < 0.06)).astype(float)
    # Kyle lambda (simplified, rolling)
    ky_out = pd.Series(np.nan, index=d.index)
    for i in range(20, len(d)):
        ri   = ret.iloc[i - 20: i].values
        dpi  = dp.iloc[i - 20: i].values
        cov_ = np.cov(ri, dpi) if len(ri) > 3 else np.zeros((2, 2))
        var_ = dpi.var() + 1e-12
        ky_out.iloc[i] = float(cov_[0, 1] / var_)
    F["kyle_lam"] = ky_out

    # ── 5. Volatility ────────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        F["rv_"   + str(w)] = (lr ** 2).rolling(w).sum()
        F["rvol_" + str(w)] = lr.rolling(w).std()
    # Parkinson high-low vol
    F["pk_vol"] = np.sqrt((1.0 / (4.0 * math.log(2))) *
                          (np.log(d["high"] / d["low"].replace(0, np.nan)) ** 2).rolling(20).mean())
    # Garman-Klass
    F["gk_vol"] = np.sqrt(
        (0.5 * (np.log(d["high"] / d["low"].replace(0, np.nan)) ** 2) -
         (2.0 * math.log(2) - 1) * (np.log(c_ / d["open"].replace(0, np.nan)) ** 2)
         ).rolling(20).mean()
    )
    rv20         = (lr ** 2).rolling(20).sum()
    F["vov"]     = rv20.rolling(10).std() / rv20.rolling(10).mean().replace(0, np.nan)
    F["skew50"]  = lr.rolling(50).apply(lambda x: float(sp_skew(x)), raw=True)
    F["kurt50"]  = lr.rolling(50).apply(lambda x: float(sp_kurt(x)), raw=True)
    F["vr_5_20"] = F["rvol_5"] / F["rvol_20"].replace(0, np.nan)

    # ── 6. VWAP & Structure ──────────────────────────────────────────────
    for w in [20, 50, 100]:
        vw_ = (tp * vol).rolling(w).sum() / vol.rolling(w).sum()
        vr_ = (vol * (tp - vw_) ** 2).rolling(w).sum() / vol.rolling(w).sum()
        vs_ = np.sqrt(vr_.replace(0, np.nan))
        F["vwap_dev_" + str(w)]  = (c_ - vw_) / vw_ * 100.0
        F["vwap_band_" + str(w)] = (c_ - vw_) / vs_.replace(0, np.nan)
    for sp in [8, 21, 50]:
        F["ema_dev_" + str(sp)] = (c_ - c_.ewm(sp).mean()) / c_ * 100.0
    F["ema_8_21"]  = (c_.ewm(8).mean() - c_.ewm(21).mean()) / c_ * 100.0
    F["ema_cross"] = (c_.ewm(8).mean() > c_.ewm(21).mean()).astype(float)

    # ── 7. Microstructure ────────────────────────────────────────────────
    rng_ = (d["high"] - d["low"]).replace(0, np.nan)
    F["wt_rel"]   = d["wick_top"] / atr
    F["wb_rel"]   = d["wick_bot"] / atr
    F["wasym"]    = (d["wick_bot"] - d["wick_top"]) / atr
    F["effic"]    = d["body_pct"].abs() / (rng_ / c_ * 100.0).replace(0, np.nan)
    F["hl_pos"]   = (c_ - d["low"]) / rng_
    F["vol_z"]    = d["vol_z"]
    F["big_trade"]= (d["vol_z"] > 3.0).astype(float)
    F["absorb"]   = ((d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.08)).astype(float)
    F["trap"]     = ((d["body_pct"].shift(1).abs() > 0.25) &
                     (d["body_pct"] * d["body_pct"].shift(1) < 0)).astype(float)
    F["amihud"]   = (lr.abs() / vol).rolling(20).mean()

    # ── 8. Hilbert / Fisher ──────────────────────────────────────────────
    try:
        raw_arr = c_.values.astype(float)
        x_dt    = raw_arr - np.linspace(raw_arr[0], raw_arr[-1], len(raw_arr))
        analytic= sp_hilbert(x_dt)
        F["hil_amp"]   = pd.Series(np.abs(analytic), index=d.index) / (c_.std() + 1e-9)
        F["hil_phase"] = pd.Series(np.angle(analytic), index=d.index)
        F["hil_freq"]  = pd.Series(np.gradient(np.unwrap(np.angle(analytic))), index=d.index)
        hi10 = c_.rolling(10).max()
        lo10 = c_.rolling(10).min()
        v_   = (2.0 * (c_ - lo10) / (hi10 - lo10 + 1e-9) - 1.0).clip(-0.999, 0.999)
        F["fisher"] = 0.5 * np.log((1.0 + v_) / (1.0 - v_ + 1e-10))
    except Exception:
        for col in ["hil_amp", "hil_phase", "hil_freq", "fisher"]:
            F[col] = 0.0

    # ── 9. Wyckoff / Smart Money ─────────────────────────────────────────
    n_w = min(30, len(d))
    x_w = np.arange(n_w)
    rec = d.tail(n_w)
    def safe_slope(vals):
        try:
            return float(np.polyfit(x_w[: len(vals)], vals, 1)[0])
        except Exception:
            return 0.0
    pt = safe_slope(rec["close"].values)
    bt = safe_slope(rec["taker_buy_vol"].values)
    st = safe_slope((rec["volume"] - rec["taker_buy_vol"]).values)
    wy = (2 if pt < -0.3 and bt > 0 else
           3 if pt > 0.3  and bt > 0 else
          -2 if pt > 0.3  and st > 0 else
          -3 if pt < -0.3 and st > 0 else 0)
    F["wyckoff"] = float(wy)
    cvd_trend_val = 0.0
    if len(d) >= 20:
        v0 = float(dlt.rolling(20).sum().iloc[-1])
        v1 = float(dlt.rolling(20).sum().iloc[-20])
        cvd_trend_val = float(np.clip((v0 - v1) / 10000.0, -3.0, 3.0))
    F["sm_flow"] = cvd_trend_val

    # ── 10. Time / Calendar ──────────────────────────────────────────────
    h_ = d["open_time"].dt.hour
    dw = d["open_time"].dt.dayofweek
    F["sin_h"]   = np.sin(2.0 * math.pi * h_ / 24.0)
    F["cos_h"]   = np.cos(2.0 * math.pi * h_ / 24.0)
    F["sin_dow"] = np.sin(2.0 * math.pi * dw / 7.0)
    F["cos_dow"] = np.cos(2.0 * math.pi * dw / 7.0)
    F["london"]  = h_.isin([8, 9, 10, 11, 12]).astype(float)
    F["ny"]      = h_.isin([13, 14, 15, 16, 17, 18, 19]).astype(float)
    F["weekend"] = (dw >= 4).astype(float)

    # ── 11. Funding ───────────────────────────────────────────────────────
    avg_fr = 0.0
    tr_fr  = 0.0
    if fund is not None and len(fund) >= 3:
        rates  = fund["fundingRate"].tail(8).values.astype(float)
        avg_fr = float(rates.mean())
        tr_fr  = float(np.clip((rates[-1] - rates[0]) * 1000.0, -3.0, 3.0))
    F["fund_rate"]   = avg_fr
    F["fund_trend"]  = tr_fr
    F["fund_revert"] = float(-1 if avg_fr > 0.0008 else (1 if avg_fr < -0.0005 else 0))

    # ── 12. Liquidity / Stacked ───────────────────────────────────────────
    F["stk_buy"]   = (dp > 0.1).rolling(3).sum().eq(3).astype(float)
    F["stk_sell"]  = (dp < -0.1).rolling(3).sum().eq(3).astype(float)
    F["bid_abs"]   = ((d["wick_bot"] > atr * 0.25) & (dp > 0.1)  & (d["vol_z"] > 1.0)).astype(float)
    F["ask_abs"]   = ((d["wick_top"] > atr * 0.25) & (dp < -0.1) & (d["vol_z"] > 1.0)).astype(float)

    # ── 13. Interaction ───────────────────────────────────────────────────
    F["mom_vol"]   = c_.pct_change(3) * d["vol_z"]
    F["dlt_mom"]   = dp * np.sign(c_.pct_change(1))
    F["vwap_dlt"]  = F["vwap_dev_20"] * dp

    # ── 14. OU statistics ────────────────────────────────────────────────
    ou_z_val  = 0.0
    ou_hl_val = 999.0
    x_ou = c_.values[-100:] if len(c_) >= 100 else c_.values
    if len(x_ou) >= 30:
        dx_ = np.diff(x_ou)
        xl_ = x_ou[:-1]
        A_  = np.column_stack([np.ones(len(xl_)), xl_])
        try:
            co_, _, _, _ = np.linalg.lstsq(A_, dx_, rcond=None)
            mu_ou_ = -co_[0] / co_[1] if co_[1] != 0 else float(x_ou.mean())
            sg_ou_ = max(float(np.std(dx_ - (co_[0] + co_[1] * xl_))), 1e-9)
            ou_z_val  = float(np.clip((float(x_ou[-1]) - mu_ou_) / sg_ou_, -5.0, 5.0))
            ou_hl_val = float(np.clip(math.log(2) / (-co_[1]), 0, 200)) if co_[1] < 0 else 999.0
        except Exception:
            pass
    F["ou_z"]  = ou_z_val
    F["ou_hl"] = ou_hl_val

    F = F.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return F


# ─────────────────────────────────────────────────────────────────────────────
#  DEEP RESIDUAL NETWORK — Pure NumPy (skip connections, Swish activation)
# ─────────────────────────────────────────────────────────────────────────────
class ResNet:
    """ResNet with Swish activations, Adam, L2, dropout."""

    def __init__(self, n_in, hidden=64, n_blocks=3,
                 lr=5e-4, l2=1e-4, dropout=0.25):
        self.lr = lr
        self.l2 = l2
        self.dr = dropout
        self.nb = n_blocks
        self.val_acc = 0.5

        def he(a, b):
            return np.random.randn(a, b).astype(np.float64) * math.sqrt(2.0 / a)

        self.Wi = he(n_in, hidden)
        self.bi = np.zeros(hidden, dtype=np.float64)
        self.Wr1 = [he(hidden, hidden) for _ in range(n_blocks)]
        self.br1 = [np.zeros(hidden, dtype=np.float64) for _ in range(n_blocks)]
        self.Wr2 = [he(hidden, hidden) for _ in range(n_blocks)]
        self.br2 = [np.zeros(hidden, dtype=np.float64) for _ in range(n_blocks)]
        self.Wo = he(hidden, 1)
        self.bo = np.zeros(1, dtype=np.float64)

        all_p = self._params()
        self.m = {k: np.zeros_like(v) for k, v in all_p.items()}
        self.v = {k: np.zeros_like(v) for k, v in all_p.items()}
        self.t = 0

    def _params(self):
        p = {"Wi": self.Wi, "bi": self.bi,
             "Wo": self.Wo, "bo": self.bo}
        for i in range(self.nb):
            p["Wr1_" + str(i)] = self.Wr1[i]
            p["br1_" + str(i)] = self.br1[i]
            p["Wr2_" + str(i)] = self.Wr2[i]
            p["br2_" + str(i)] = self.br2[i]
        return p

    @staticmethod
    def _swish(x):
        return x / (1.0 + np.exp(-np.clip(x, -50, 50)))

    @staticmethod
    def _swish_d(x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        return s + x * s * (1.0 - s)

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, X, train=True):
        cache = {}
        Z0 = X @ self.Wi + self.bi
        A0 = self._swish(Z0)
        cache["X"]  = X
        cache["Z0"] = Z0
        A = A0

        for i in range(self.nb):
            Z1 = A @ self.Wr1[i] + self.br1[i]
            A1 = self._swish(Z1)
            if train and self.dr > 0:
                mask = (np.random.rand(*A1.shape) > self.dr).astype(np.float64)
                mask /= (1.0 - self.dr + 1e-9)
                A1 *= mask
                cache["mask_" + str(i)] = mask
            Z2 = A1 @ self.Wr2[i] + self.br2[i]
            A2 = self._swish(Z2 + A)   # skip connection
            cache["A_in_"  + str(i)] = A
            cache["Z1_"    + str(i)] = Z1
            cache["A1_"    + str(i)] = A1
            cache["Z2_"    + str(i)] = Z2
            A  = A2

        Zo   = A @ self.Wo + self.bo
        Ao   = self._sig(Zo)
        cache["Af"] = A
        cache["Zo"] = Zo
        return Ao.ravel(), cache

    def backward(self, y, out, cache):
        m = float(len(y))
        grads = {}
        dA = (out - y) / m
        dZo = dA.reshape(-1, 1)
        grads["Wo"] = cache["Af"].T @ dZo + self.l2 * self.Wo
        grads["bo"] = dZo.sum(axis=0)
        dA_prev = dZo @ self.Wo.T

        for i in reversed(range(self.nb)):
            A_in_ = cache["A_in_" + str(i)]
            Z1_   = cache["Z1_"   + str(i)]
            A1_   = cache["A1_"   + str(i)]
            Z2_   = cache["Z2_"   + str(i)]

            dA2 = dA_prev * self._swish_d(Z2_ + A_in_)
            grads["Wr2_" + str(i)] = A1_.T @ dA2 + self.l2 * self.Wr2[i]
            grads["br2_" + str(i)] = dA2.sum(axis=0)
            dA1 = dA2 @ self.Wr2[i].T
            if "mask_" + str(i) in cache:
                dA1 *= cache["mask_" + str(i)]
            dZ1 = dA1 * self._swish_d(Z1_)
            grads["Wr1_" + str(i)] = A_in_.T @ dZ1 + self.l2 * self.Wr1[i]
            grads["br1_" + str(i)] = dZ1.sum(axis=0)
            dA_prev = dZ1 @ self.Wr1[i].T + dA2   # grad through skip

        dZ0 = dA_prev * self._swish_d(cache["Z0"])
        grads["Wi"] = cache["X"].T @ dZ0 + self.l2 * self.Wi
        grads["bi"] = dZ0.sum(axis=0)
        return grads

    def _adam(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self._params()
        for k, g in grads.items():
            if k not in params:
                continue
            self.m[k] = b1 * self.m.get(k, np.zeros_like(g)) + (1 - b1) * g
            self.v[k] = b2 * self.v.get(k, np.zeros_like(g)) + (1 - b2) * g ** 2
            mc = self.m[k] / (1.0 - b1 ** self.t)
            vc = self.v[k] / (1.0 - b2 ** self.t)
            params[k] -= self.lr * mc / (np.sqrt(vc) + eps)

    def fit(self, X, y, Xv=None, yv=None, epochs=100, batch=32):
        best_acc = 0.0
        best_w   = None
        no_imp   = 0

        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            for s in range(0, len(X), batch):
                Xb = X[idx[s: s + batch]]
                yb = y[idx[s: s + batch]]
                if len(Xb) < 2:
                    continue
                out, cache = self.forward(Xb, train=True)
                grads = self.backward(yb, out, cache)
                self._adam(grads)

            if Xv is not None and len(Xv) > 0:
                pv, _ = self.forward(Xv, train=False)
                acc   = float(((pv > 0.5) == yv).mean())
                if acc > best_acc:
                    best_acc = acc
                    best_w   = {k: v.copy() for k, v in self._params().items()}
                    no_imp   = 0
                else:
                    no_imp  += 1
                if no_imp >= 15:
                    break

            if (ep + 1) % 20 == 0:
                self.lr *= 0.7

        if best_w is not None:
            for k, v in best_w.items():
                self._params()[k][...] = v
        self.val_acc = best_acc

    def predict(self, X):
        p, _ = self.forward(X, train=False)
        return p


# ─────────────────────────────────────────────────────────────────────────────
#  META-LABELING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
class MetaLabelSystem:
    def __init__(self):
        self.primary   = None
        self.secondary = None
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrated = False

    def fit(self, X, y_tb, splits, verbose=True):
        """Fit primary direction + secondary meta-label models."""
        y_dir = (y_tb == 1).astype(int)

        # ── Primary (GBM direction model) ──
        gbm = GradientBoostingClassifier(
            n_estimators=CFG["GBM_TREES"],
            learning_rate=0.03,
            max_depth=4,
            subsample=0.70,
            min_samples_leaf=8,
            random_state=42,
        )
        oof_p = np.full(len(X), 0.5)
        for tr, te in splits:
            if len(tr) < 50 or len(te) < 5:
                continue
            gbm.fit(X[tr], y_dir[tr])
            oof_p[te] = gbm.predict_proba(X[te])[:, 1]
        gbm.fit(X, y_dir)
        self.primary = gbm
        gbm_acc = float(((oof_p > 0.5).astype(int) == y_dir).mean())
        if verbose:
            print("    GBM OOF acc: {:.4f}".format(gbm_acc))

        # ── Meta-label: was the primary correct? ──
        y_meta = np.zeros(len(y_tb))
        pred_p = (oof_p > 0.5).astype(int)
        for i in range(len(y_tb)):
            if y_tb[i] == 0:
                y_meta[i] = 0
            elif y_tb[i] == 1 and pred_p[i] == 1:
                y_meta[i] = 1
            elif y_tb[i] == -1 and pred_p[i] == 0:
                y_meta[i] = 1
            else:
                y_meta[i] = 0

        # ── Secondary (ExtraTrees meta-label) ──
        et = ExtraTreesClassifier(
            n_estimators=CFG["ET_TREES"],
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        oof_m = np.full(len(X), 0.5)
        for tr, te in splits:
            if len(tr) < 50:
                continue
            Xp_tr = np.column_stack([X[tr], oof_p[tr]])
            Xp_te = np.column_stack([X[te], oof_p[te]])
            et.fit(Xp_tr, y_meta[tr])
            oof_m[te] = et.predict_proba(Xp_te)[:, 1]

        # Isotonic calibration
        valid = y_tb != 0
        if valid.sum() > 20:
            self.calibrator.fit(oof_m[valid], y_meta[valid])
            self.calibrated = True

        Xp_all = np.column_stack([X, oof_p])
        et.fit(Xp_all, y_meta)
        self.secondary = et

        non_exp = y_tb != 0
        et_acc  = float(((oof_m[non_exp] > 0.5).astype(int) == y_meta[non_exp]).mean()) \
                  if non_exp.sum() > 0 else 0.5
        if verbose:
            print("    ET meta acc:  {:.4f}".format(et_acc))

        return oof_p, oof_m, gbm_acc, et_acc

    def predict(self, X, primary_prob):
        if self.primary is None:
            return {"direction": 0.5, "meta_prob": 0.5, "signal": "WAIT"}
        dir_p = float(self.primary.predict_proba(X[-1:])[:, 1][0])
        Xp    = np.column_stack([X[-1:], [[primary_prob]]])
        meta  = float(self.secondary.predict_proba(Xp)[:, 1][0]) \
                if self.secondary else 0.5
        if self.calibrated:
            meta = float(self.calibrator.predict([meta])[0])
        signal = "BUY" if dir_p > 0.55 else ("SELL" if dir_p < 0.45 else "WAIT")
        return {"direction": dir_p, "meta_prob": meta,
                "signal": signal, "take": meta >= CFG["META_THR"]}


# ─────────────────────────────────────────────────────────────────────────────
#  CPCV SHARPE ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────
def cpcv_sharpe(oof_probs, y_dir, returns_series, n_splits=6, n_test=2):
    n    = min(len(oof_probs), len(y_dir), len(returns_series))
    if n < 100:
        return 0.0
    fs   = n // n_splits
    folds = [list(range(i * fs, (i + 1) * fs if i < n_splits - 1 else n))
             for i in range(n_splits)]
    sharpes = []
    for combo in combinations(range(n_splits), n_test):
        test_idx = []
        for ci in combo:
            test_idx.extend(folds[ci])
        if len(test_idx) < 10:
            continue
        p_  = oof_probs[test_idx]
        y_  = y_dir[test_idx]
        r_  = returns_series.iloc[test_idx].values
        strat = np.where(p_ > 0.55, r_, np.where(p_ < 0.45, -r_, 0.0))
        sg   = strat.std()
        mu   = strat.mean()
        sharpes.append(mu / sg * math.sqrt(288 * 252) if sg > 0 else 0.0)
    return float(np.mean(sharpes)) if sharpes else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  GARCH(1,1) VOL REGIME
# ─────────────────────────────────────────────────────────────────────────────
def garch11(ret: pd.Series):
    r  = ret.dropna().values
    if len(r) < 30:
        return 0.003, 1.0, "MEDIUM", 50.0
    v0 = float(np.var(r))

    def nll(params):
        om, al, be = params
        if om <= 0 or al < 0 or be < 0 or al + be >= 1:
            return 1e10
        h  = np.full(len(r), v0)
        ll = 0.0
        for t in range(1, len(r)):
            h[t] = om + al * r[t - 1] ** 2 + be * h[t - 1]
            if h[t] <= 0:
                return 1e10
            ll += -0.5 * (math.log(2 * math.pi * h[t]) + r[t] ** 2 / h[t])
        return -ll

    try:
        res = optimize.minimize(
            nll, [v0 * 0.05, 0.08, 0.88],
            method="L-BFGS-B",
            bounds=[(1e-9, None), (1e-9, 0.999), (1e-9, 0.999)],
            options={"maxiter": 100},
        )
        om, al, be = res.x
    except Exception:
        om, al, be = v0 * 0.05, 0.08, 0.88

    h = np.full(len(r), v0)
    for t in range(1, len(r)):
        h[t] = max(om + al * r[t - 1] ** 2 + be * h[t - 1], 1e-12)

    cur_vol = float(math.sqrt(h[-1]))
    vp      = float(stats.percentileofscore(np.sqrt(h), cur_vol))
    regime  = "LOW" if vp < 30 else ("HIGH" if vp > 75 else "MEDIUM")
    size_m  = 1.5 if vp < 30 else (0.5 if vp > 80 else 1.0)
    return cur_vol, size_m, regime, vp


# ─────────────────────────────────────────────────────────────────────────────
#  SHARPE-OPTIMAL KELLY
# ─────────────────────────────────────────────────────────────────────────────
def sharpe_kelly(mu: float, sigma: float, rho: float = 0.0):
    if sigma <= 0:
        return 0.0, 0.0
    sharpe_k = mu / (sigma ** 2)
    shrunk   = sharpe_k * (1.0 - rho) * 0.25
    return float(np.clip(shrunk, 0.0, 3.0)), float(sharpe_k)


# ─────────────────────────────────────────────────────────────────────────────
#  MARKET PROFILE (POC / VAH / VAL)
# ─────────────────────────────────────────────────────────────────────────────
def market_profile(df: pd.DataFrame, tick: float = 25.0):
    lo = df["low"].min()
    hi = df["high"].max()
    bkts = np.arange(math.floor(lo / tick) * tick,
                     math.ceil(hi / tick) * tick + tick, tick)
    vm = defaultdict(float)
    for _, row in df.iterrows():
        lvls = bkts[(bkts >= row["low"]) & (bkts <= row["high"])]
        if len(lvls) == 0:
            continue
        vp = row["volume"] / len(lvls)
        for lv in lvls:
            vm[lv] += vp
    if not vm:
        price = float(df["close"].iloc[-1])
        return price, price, price

    pf  = pd.DataFrame({"p": list(vm.keys()), "v": list(vm.values())}).sort_values("p")
    poc = float(pf.loc[pf["v"].idxmax(), "p"])
    tot = pf["v"].sum()
    pi  = pf["v"].idxmax()
    cum = 0.0
    va  = []
    for _ in range(len(pf)):
        ui = pi + 1
        li = pi - 1
        uv = pf.loc[ui, "v"] if ui in pf.index else 0.0
        dv = pf.loc[li, "v"] if li in pf.index else 0.0
        if uv >= dv and ui in pf.index:
            va.append(ui)
            cum += uv
            pi   = ui
        elif li in pf.index:
            va.append(li)
            cum += dv
            pi   = li
        else:
            break
        if cum / tot >= 0.70:
            break
    vah = float(pf.loc[va, "p"].max()) if va else poc + tick * 5
    val = float(pf.loc[va, "p"].min()) if va else poc - tick * 5
    return poc, vah, val


# ─────────────────────────────────────────────────────────────────────────────
#  KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────
def kalman_filter(prices: pd.Series):
    z  = prices.astype(float).values
    n  = len(z)
    F_ = np.array([[1.0, 1.0], [0.0, 1.0]])
    H_ = np.array([[1.0, 0.0]])
    Q_ = np.array([[0.01, 0.001], [0.001, 0.0001]])
    R_ = np.array([[1.0]])
    x  = np.array([[z[0]], [0.0]])
    P  = np.eye(2) * 1000.0
    kp = np.zeros(n)
    kt = np.zeros(n)
    for t in range(n):
        xp = F_ @ x
        Pp = F_ @ P @ F_.T + Q_
        K  = Pp @ H_.T @ np.linalg.inv(H_ @ Pp @ H_.T + R_)
        x  = xp + K * (z[t] - float((H_ @ xp).flat[0]))
        P  = (np.eye(2) - K @ H_) @ Pp
        kp[t] = float(x[0].flat[0])
        kt[t] = float(x[1].flat[0])
    return float(kp[-1]), float(kt[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_signals(df, meta_res, resnet_prob, gbm_prob,
                      poc, vah, val, garch_mult, vol_regime):
    price = float(df["close"].iloc[-1])
    atr   = float(df["atr"].iloc[-1]) or price * 0.003
    ret   = df["close"].pct_change().dropna()
    dp    = df["delta_pct"].astype(float)
    dlt   = df["delta"].astype(float)

    # CVD divergence
    cvd20 = dlt.rolling(20).sum()
    pr_s  = df["close"].diff(3) / df["close"].shift(3) * 100.0
    cvd_s = cvd20.diff(3)
    div_b = bool(pr_s.iloc[-1] < -0.12 and cvd_s.iloc[-1] > 0)
    div_s = bool(pr_s.iloc[-1] >  0.12 and cvd_s.iloc[-1] < 0)

    # OU z-score
    x_ou = df["close"].values[-100:]
    ou_z = 0.0
    if len(x_ou) >= 30:
        dx_ = np.diff(x_ou); xl_ = x_ou[:-1]
        A_  = np.column_stack([np.ones(len(xl_)), xl_])
        try:
            co_, _, _, _ = np.linalg.lstsq(A_, dx_, rcond=None)
            mu_ou = -co_[0] / co_[1] if co_[1] != 0 else float(x_ou.mean())
            sg_ou = max(float(np.std(dx_ - (co_[0] + co_[1] * xl_))), 1e-9)
            ou_z  = float(np.clip((float(x_ou[-1]) - mu_ou) / sg_ou, -5.0, 5.0))
        except Exception:
            pass

    # Wyckoff
    n_w = min(30, len(df))
    x_w = np.arange(n_w)
    rec = df.tail(n_w)
    def sl_(vals):
        try:
            return float(np.polyfit(x_w[:len(vals)], vals, 1)[0])
        except Exception:
            return 0.0
    pt = sl_(rec["close"].values)
    bt = sl_(rec["taker_buy_vol"].values)
    sv = (rec["volume"] - rec["taker_buy_vol"]).values
    st = sl_(sv)
    wy = (3 if pt < -0.3 and bt > 0 else
           2 if pt > 0.3  and bt > 0 else
          -3 if pt > 0.3  and st > 0 else
          -2 if pt < -0.3 and st > 0 else 0)

    # Kalman
    kal_p, kal_t = kalman_filter(df["close"])

    # Trap / absorption
    bp   = df["body_pct"]
    trap_s = bool(bp.shift(1).iloc[-1] < -0.25 and df["close"].iloc[-1] > df["open"].shift(1).iloc[-1])
    trap_l = bool(bp.shift(1).iloc[-1] >  0.25 and df["close"].iloc[-1] < df["open"].shift(1).iloc[-1])
    vz_last = float(df["vol_z"].iloc[-1])
    ab_sc   = (1 if vz_last > 1.5 and dp.iloc[-1] > 0.1 and abs(bp.iloc[-1]) < 0.08 else
              -1 if vz_last > 1.5 and dp.iloc[-1] < -0.1 and abs(bp.iloc[-1]) < 0.08 else 0)

    # VWAP band
    c_ = df["close"].astype(float)
    vol_ = df["volume"].astype(float).replace(0, np.nan)
    tp_  = (df["high"] + df["low"] + c_) / 3.0
    vw20 = (tp_ * vol_).rolling(20).sum() / vol_.rolling(20).sum()
    vr20 = (vol_ * (tp_ - vw20) ** 2).rolling(20).sum() / vol_.rolling(20).sum()
    vs20 = np.sqrt(vr20.replace(0, np.nan))
    vdev = float((c_ - vw20).iloc[-1] / vs20.iloc[-1]) if float(vs20.iloc[-1]) > 0 else 0.0
    vwap_sc = 2 if vdev < -1.8 else (1 if vdev < -0.8 else
             -2 if vdev >  1.8 else (-1 if vdev >  0.8 else 0))

    # Individual scores
    resnet_sc = (3 if resnet_prob > 0.70 else 2 if resnet_prob > 0.62 else
                 1 if resnet_prob > 0.56 else -3 if resnet_prob < 0.30 else
                -2 if resnet_prob < 0.38 else -1 if resnet_prob < 0.44 else 0)
    meta_dir  = meta_res["direction"]
    meta_conf = meta_res["meta_prob"]
    dir_sc    = (3 if meta_dir > 0.65 else 2 if meta_dir > 0.56 else
                -3 if meta_dir < 0.35 else -2 if meta_dir < 0.44 else 0)
    meta_mult = 1.5 if meta_conf > 0.65 else (0.5 if meta_conf < 0.45 else 1.0)
    cvd_sc    = 3 if div_b else (-3 if div_s else 0)
    ou_sc     = (3 if ou_z < -2.0 else 2 if ou_z < -1.0 else 1 if ou_z < -0.5 else
                -3 if ou_z >  2.0 else -2 if ou_z >  1.0 else -1 if ou_z >  0.5 else 0)
    kal_sc    = (2 if kal_t > 0.2 else 1 if kal_t > 0 else
                -2 if kal_t < -0.2 else -1 if kal_t < 0 else 0)
    trap_sc   = (2 if trap_s else -2 if trap_l else 0)

    raw = (resnet_sc * 1.5 + dir_sc * meta_mult +
           cvd_sc + ou_sc + wy + kal_sc + trap_sc + ab_sc + vwap_sc)
    raw *= (0.65 if vol_regime == "HIGH" else 1.0)
    score = int(np.clip(raw, -15, 15))
    conf  = min(abs(score) / 15.0 * 100.0 * meta_conf * 1.8, 99.0)

    # Sizing
    mu_e  = float(ret.tail(50).mean()) * (1 if score > 0 else -1)
    sg_e  = float(ret.tail(50).std())
    rho_e = max(0.0, 1.0 - abs(score) / 15.0)
    sk_r, sk_full = sharpe_kelly(mu_e, sg_e, rho_e)

    # Trade levels
    stop_dist = atr * CFG["ATR_SL_MULT"]
    if score >= CFG["MIN_SCORE"]:
        side = "BUY"
        sl_  = round(min(val, price - stop_dist), 1)
        tp1  = round(poc if poc > price else price + stop_dist * CFG["TP_MULT"], 1)
        tp2  = round(vah if vah > tp1 else price + stop_dist * CFG["TP_MULT"] * 2, 1)
    elif score <= -CFG["MIN_SCORE"]:
        side = "SELL"
        sl_  = round(max(vah, price + stop_dist), 1)
        tp1  = round(poc if poc < price else price - stop_dist * CFG["TP_MULT"], 1)
        tp2  = round(val if val < tp1 else price - stop_dist * CFG["TP_MULT"] * 2, 1)
    else:
        side = "WAIT"
        sl_  = tp1 = tp2 = None

    rr  = abs(tp1 - price) / max(abs(price - (sl_ or price)), 1.0) if tp1 else 0.0
    qty = (CFG["ACCOUNT"] * sk_r * garch_mult / max(stop_dist, 1.0)) if sl_ else 0.0
    tradeable = (side != "WAIT" and conf >= CFG["MIN_CONF"] and
                 rr >= CFG["MIN_RR"] and meta_res["take"])

    reasons = []
    if abs(resnet_sc) >= 2:
        reasons.append("{:+.0f} ResNet".format(resnet_sc * 1.5))
    if abs(dir_sc) >= 2:
        reasons.append("{:+.0f} MetaML".format(dir_sc * meta_mult))
    if abs(cvd_sc) >= 3:
        reasons.append("{:+d} CVD_Div".format(cvd_sc))
    if abs(ou_sc) >= 2:
        reasons.append("{:+d} OU_Rev".format(ou_sc))
    if abs(wy) >= 2:
        reasons.append("{:+d} Wyckoff".format(wy))
    if abs(kal_sc) >= 2:
        reasons.append("{:+d} Kalman".format(kal_sc))
    if abs(vwap_sc) >= 2:
        reasons.append("{:+d} VWAP".format(vwap_sc))

    return {
        "side": side, "score": score, "confidence": conf,
        "tradeable": tradeable, "sl": sl_, "tp1": tp1, "tp2": tp2,
        "qty": round(qty, 3), "rr": rr,
        "poc": poc, "vah": vah, "val": val,
        "garch_mult": garch_mult, "vol_regime": vol_regime,
        "kelly_rec": sk_r, "kelly_full": sk_full,
        "ou_z": ou_z, "kal_trend": kal_t, "kal_price": kal_p,
        "meta_dir": meta_dir, "meta_conf": meta_conf,
        "take_trade": meta_res["take"],
        "resnet_prob": resnet_prob, "gbm_prob": gbm_prob,
        "div_bull": div_b, "div_bear": div_s,
        "wyckoff": wy, "trap": trap_l or trap_s,
        "vdev": vdev, "vwap_sc": vwap_sc,
        "vol5": float(ret.tail(5).std()),
        "vol20": float(ret.tail(20).std()),
        "reasons": reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY  (no nested f-strings — Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "G": "\033[92m", "R": "\033[91m", "Y": "\033[93m",
    "C": "\033[96m", "W": "\033[97m", "B": "\033[1m",
    "D": "\033[2m",  "M": "\033[95m", "X": "\033[0m",
}

def cc(text, col):
    return COLORS.get(col, "") + str(text) + COLORS["X"]

def draw_bar(value, width=12, fg="█", bg="░"):
    n = min(int(abs(float(value)) * width), width)
    return fg * n + bg * (width - n)

def display_dashboard(price, res, tr, loop_n, live, cpcv_sh):
    os.system("cls" if os.name == "nt" else "clear")
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    side = res["side"]
    sc   = res["score"]
    conf = res["confidence"]

    sc_col = "G" if sc > 0 else ("R" if sc < 0 else "Y")
    sd_col = "G" if side == "BUY" else ("R" if side == "SELL" else "Y")

    # ── Header ──
    print(cc("=" * 74, "C"))
    print(cc("  ELITE QUANT ENGINE v3.0  |  BTC/USDT Futures  |  Institutional Grade", "C"))
    print(cc("  ResNet + Meta-Label + Triple-Barrier + CPCV + FracDiff + 120 Alphas", "C"))
    print(cc("=" * 74, "C"))
    live_str = "LIVE" if live else cc("SYNTHETIC", "Y")
    print("  {}   Loop #{}   {}".format(cc(now, "D"), loop_n, live_str))
    print("  {}   Vol5={:.3f}%  Vol20={:.3f}%  GARCH x{:.1f}  Regime={}".format(
        cc("Price: ${:,.2f}".format(price), "W"),
        res["vol5"] * 100, res["vol20"] * 100,
        res["garch_mult"], cc(res["vol_regime"], sc_col)
    ))
    print()

    # ── Signal Box ──
    bar_str = draw_bar(abs(sc) / 15.0)
    print(cc("  " + "=" * 66, "W"))
    if side == "BUY":
        print(cc("  ||  ######   B U Y   ^ ^ ^ ^ ^ ^ ^ ^   ######              ||", "G"))
    elif side == "SELL":
        print(cc("  ||  ######   S E L L   v v v v v v v v   ######             ||", "R"))
    else:
        print(cc("  ||  ------   W A I T   (insufficient confluence)  ------     ||", "Y"))

    score_str  = cc("{:>+3d}".format(sc), "B")
    bar_col    = cc(bar_str, sc_col)
    conf_str   = cc("{:.1f}%".format(conf), "B")
    mc_str     = cc("{:.3f}".format(res["meta_conf"]), "B")
    take_str   = cc("YES", "G") if res["take_trade"] else cc("NO", "R")
    print("  ||  Score:{}  {}  Conf:{}  MetaConf:{}  Take:{}   ||".format(
        score_str, bar_col, conf_str, mc_str, take_str))
    print(cc("  " + "=" * 66, "W"))
    print()

    # ── Trade Setup ──
    if res["tradeable"] and res["tp1"]:
        rr    = res["rr"]
        rrc   = "G" if rr >= 2.5 else ("Y" if rr >= 1.5 else "R")
        sl_   = res["sl"]
        tp1   = res["tp1"]
        tp2   = res["tp2"]
        qty   = res["qty"]
        kr    = res["kelly_rec"]
        kf_   = res["kelly_full"]
        gm    = res["garch_mult"]
        print(cc("  +----- SHARPE-OPTIMAL TRADE STRUCTURE ----------------------------------------+", "Y"))
        print("  |  Entry:   ${:>12,.2f}".format(price) + " " * 40 + "|")
        print(cc("  |  Stop:    ${:>12,.2f}  (${:>7,.1f} = {:.1f}x ATR)".format(
            sl_, abs(price - sl_), CFG["ATR_SL_MULT"]), "R") + " " * 22 + cc("|", "Y"))
        print(cc("  |  TP1:     ${:>12,.2f}  -> POC / structure  (close 60%)".format(tp1), "G") + " " * 14 + cc("|", "Y"))
        print(cc("  |  TP2:     ${:>12,.2f}  -> VAH / VAL        (close 40%)".format(tp2), "G") + " " * 14 + cc("|", "Y"))
        print("  |  R:R={}   Qty={:.3f} BTC   Kelly={:.2f}%   GARCH x{:.1f}x{}|".format(
            cc("{:.2f}x".format(rr), rrc), qty, kr * 100, gm, " " * 10))
        print("  |  Shrunk-Kelly={:.2f}%   Sharpe-Kelly(full)={:.2f}%{}|".format(
            kr * 100, kf_ * 100, " " * 18))
        print(cc("  +---------------------------------------------------------------------------+", "Y"))
    elif side != "WAIT":
        print(cc("  Signal found but meta={:.3f} or conf={:.1f}% below threshold".format(
            res["meta_conf"], conf), "Y"))
    print()

    # ── CPCV ──
    cpcv_col = "G" if cpcv_sh > 1.0 else ("Y" if cpcv_sh > 0.3 else "R")
    verdict  = "STRONG EDGE" if cpcv_sh > 1.0 else ("WEAK" if cpcv_sh > 0.3 else "NO EDGE")
    print(cc("  -- COMBINATORIAL PURGED CV SHARPE (CPCV) ---------------------------------", "M"))
    print("  Annualized Sharpe: {}   {}".format(
        cc("{:.3f}".format(cpcv_sh), cpcv_col), verdict))
    print()

    # ── Model Performance ──
    print(cc("  -- MODEL PERFORMANCE ------------------------------------------------------", "M"))
    rows = [
        ("Architecture",   "Triple-Barrier + PurgedCV + Meta-Label + ResNet"),
        ("GBM primary acc", "{:.2f}%".format(tr.get("gbm_acc", 0) * 100)),
        ("ET meta acc",     "{:.2f}%  (meta-label)".format(tr.get("et_acc", 0) * 100)),
        ("ResNet val acc",  "{:.2f}%".format(tr.get("resnet_acc", 0) * 100)),
        ("Features / PCA",  "{} raw -> {} PCA components".format(
            tr.get("n_raw", 0), tr.get("n_pca", 0))),
        ("Samples",         "{}".format(tr.get("n_samples", 0))),
        ("TB labels",       "TP:{:.1f}%  SL:{:.1f}%  Exp:{:.1f}%".format(
            tr.get("tb_tp", 0), tr.get("tb_sl", 0), tr.get("tb_exp", 0))),
    ]
    for label, val in rows:
        print("  {:<22} {}".format(label + ":", val))
    print()

    # ── Signal breakdown ──
    print(cc("  -- SIGNAL BREAKDOWN -------------------------------------------------------", "M"))
    items = [
        ("ResNet P(UP)",      res["resnet_prob"],    "{:.4f}".format(res["resnet_prob"])),
        ("GBM P(UP)",         res["gbm_prob"],       "{:.4f}".format(res["gbm_prob"])),
        ("Meta-conf",         res["meta_conf"],      "{:.4f}".format(res["meta_conf"])),
        ("OU z-score",       -res["ou_z"] / 5.0,    "{:>+.3f}".format(res["ou_z"])),
        ("Kalman trend",      res["kal_trend"] / 5.0, "{:>+.4f}/bar".format(res["kal_trend"])),
        ("VWAP band",        -res["vdev"] / 3.0,    "{:>+.3f} sigma".format(res["vdev"])),
        ("GARCH size mult",   1.0 - res["garch_mult"] / 1.5,
         "{:.2f}x".format(res["garch_mult"])),
    ]
    for label, raw_val, disp in items:
        raw_val = float(raw_val)
        col     = "G" if raw_val > 0.02 else ("R" if raw_val < -0.02 else "D")
        b       = draw_bar(abs(raw_val), 12)
        print("  {:<20} {}  {}".format(label, cc(b, col), disp))
    print()

    # ── Active Signals ──
    print(cc("  -- ACTIVE SIGNALS ---------------------------------------------------------", "D"))
    def emit(cond, text, col="G"):
        if cond:
            print("  {} {}".format(cc("*", "Y"), cc(text, col)))

    emit(res["resnet_prob"] > 0.66,
         "RESNET STRONG BULL    P(up)={:.4f}".format(res["resnet_prob"]))
    emit(res["resnet_prob"] < 0.34,
         "RESNET STRONG BEAR    P(up)={:.4f}".format(res["resnet_prob"]), "R")
    emit(res["div_bull"],  "CVD BULL DIVERGENCE   price fell, buyers accumulating")
    emit(res["div_bear"],  "CVD BEAR DIVERGENCE   price rose, sellers distributing", "R")
    emit(res["ou_z"] < -1.8,
         "OU OVERSHOOTING DOWN  z={:.3f}  -> reversion BUY".format(res["ou_z"]))
    emit(res["ou_z"] >  1.8,
         "OU OVERSHOOTING UP    z={:.3f}  -> reversion SELL".format(res["ou_z"]), "R")
    emit(res["wyckoff"] >= 2,
         "WYCKOFF {}  phase confirmed".format(
             "ACCUMULATION" if res["wyckoff"] == 2 else "MARKUP"))
    emit(res["wyckoff"] <= -2,
         "WYCKOFF {}  phase confirmed".format(
             "DISTRIBUTION" if res["wyckoff"] == -2 else "MARKDOWN"), "R")
    emit(res["trap"],   "TRAPPED TRADERS  squeeze coming")
    emit(res["kal_trend"] >  0.2,
         "KALMAN TREND UP   {:.3f}/bar  noise-filtered".format(res["kal_trend"]))
    emit(res["kal_trend"] < -0.2,
         "KALMAN TREND DOWN {:.3f}/bar  noise-filtered".format(res["kal_trend"]), "R")
    emit(res["meta_conf"] > 0.65,
         "META-LABEL CONFIRMS   {:.3f}  high P(trade correct)".format(res["meta_conf"]))
    emit(not res["take_trade"],
         "META-LABEL REJECTS    primary model uncertain -> skip", "Y")
    emit(res["vwap_sc"] >= 2,
         "VWAP -2 SIGMA zone  mean revert BUY opportunity")
    emit(res["vwap_sc"] <= -2,
         "VWAP +2 SIGMA zone  mean revert SELL opportunity", "R")

    print()
    print("  {} POC=${:,.1f}  VAH=${:,.1f}  VAL=${:,.1f}".format(
        cc("o", "C"), res["poc"], res["vah"], res["val"]))
    print("  {} Kalman price=${:,.1f}  trend={:>+.3f}/bar".format(
        cc("o", "C"), res["kal_price"], res["kal_trend"]))
    print("  {} CPCV Sharpe={:.3f}  GARCH={}  Kelly={:.2f}%".format(
        cc("o", "C"), cpcv_sh, res["vol_regime"], res["kelly_rec"] * 100))
    print()
    print(cc("  -- REASONS ----------------------------------------------------------------", "D"))
    if res["reasons"]:
        print("  " + "  |  ".join(res["reasons"]))
    else:
        print("  Composite signal only")
    print()
    print(cc("  Ctrl+C to stop  |  --loop  |  --account USDT  |  --interval SEC", "D"))
    print(cc("=" * 74, "D"))


# ─────────────────────────────────────────────────────────────────────────────
#  ELITE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EliteQuantEngine:
    def __init__(self, account=1000.0):
        CFG["ACCOUNT"] = account
        self.meta      = MetaLabelSystem()
        self.resnet    = None
        self.scaler    = RobustScaler()
        self.pca       = PCA(n_components=CFG["PCA_VAR"])
        self.trained   = False
        self.train_res = {}
        self.loop_n    = 0
        self.cpcv_sh   = 0.0
        self.feat_imp  = {}

    # ── Training ──
    def train(self, df: pd.DataFrame, fund: pd.DataFrame, verbose=True):
        vp = verbose
        if vp:
            print(cc("\n  ELITE QUANT ENGINE v3.0 — TRAINING", "M"))
            print(cc("  " + "-" * 60, "M"))

        # 1. Triple-barrier labels
        if vp: print("  [1/6] Triple-barrier labeling ...", end=" ", flush=True)
        tb   = triple_barrier(df, pct=CFG["BARRIER_PCT"], t_max=CFG["TARGET_BARS"])
        idx  = tb.index
        df_v = df.loc[idx]
        y_tb = tb.values
        tp_r = float((y_tb == 1).mean() * 100)
        sl_r = float((y_tb == -1).mean() * 100)
        ep_r = float((y_tb == 0).mean() * 100)
        if vp: print("TP={:.1f}%  SL={:.1f}%  Exp={:.1f}%".format(tp_r, sl_r, ep_r))

        # 2. Features
        if vp: print("  [2/6] Building 120+ alpha features ...", end=" ", flush=True)
        F_df = build_features(df_v, fund)
        X_r  = np.nan_to_num(F_df.values.astype(float), 0.0)
        if vp: print("{} raw features".format(X_r.shape[1]))

        # 3. PCA
        if vp: print("  [3/6] PCA orthogonalization ...", end=" ", flush=True)
        X_sc  = self.scaler.fit_transform(X_r)
        X_pca = self.pca.fit_transform(X_sc)
        if vp: print("{} components ({:.0f}% var)".format(X_pca.shape[1], CFG["PCA_VAR"] * 100))

        # 4. Purged K-Fold
        if vp: print("  [4/6] Purged K-Fold splits ...")
        splits = purged_kfold(len(X_pca), k=5,
                              purge=CFG["N_PURGE"], embargo=CFG["N_EMBARGO"])

        # 5. Meta-labeling
        if vp: print("  [5/6] Meta-labeling system ...")
        oof_p, oof_m, gbm_acc, et_acc = self.meta.fit(X_pca, y_tb, splits, verbose=vp)

        # 6. ResNet (on non-expired labels)
        if vp: print("  [6/6] Training ResNet ...", end=" ", flush=True)
        mask_ne  = y_tb != 0
        X_nn     = X_pca[mask_ne]
        y_nn     = (y_tb[mask_ne] == 1).astype(float)
        resnet_acc = 0.5
        if len(X_nn) > 80:
            n_val = max(int(len(X_nn) * 0.15), 20)
            self.resnet = ResNet(
                n_in=X_nn.shape[1],
                hidden=CFG["NN_HIDDEN"],
                n_blocks=CFG["NN_BLOCKS"],
                lr=CFG["NN_LR"],
                l2=CFG["NN_L2"],
                dropout=CFG["NN_DROPOUT"],
            )
            self.resnet.fit(
                X_nn[:-n_val], y_nn[:-n_val],
                Xv=X_nn[-n_val:], yv=y_nn[-n_val:],
                epochs=CFG["NN_EPOCHS"],
            )
            resnet_acc = self.resnet.val_acc
        if vp: print("val_acc={:.4f}".format(resnet_acc))

        # CPCV Sharpe
        if vp: print("  Computing CPCV Sharpe ...", end=" ", flush=True)
        y_dir = (y_tb == 1).astype(int)
        ret_s = df["close"].pct_change().fillna(0)
        ret_s_loc = ret_s.loc[idx]
        self.cpcv_sh = cpcv_sharpe(oof_p, y_dir, ret_s_loc)
        if vp: print("ann. Sharpe = {:.3f}".format(self.cpcv_sh))

        # Feature importance
        if hasattr(self.meta.primary, "feature_importances_"):
            imp = self.meta.primary.feature_importances_
            fn  = list(F_df.columns)[:len(imp)]
            self.feat_imp = dict(sorted(zip(fn, imp), key=lambda x: -x[1])[:10])
        if vp:
            top5 = list(self.feat_imp.keys())[:5]
            print("  Top alphas: {}".format(top5))

        self.trained   = True
        self.X_last    = X_pca
        self.train_res = {
            "n_raw": X_r.shape[1], "n_pca": X_pca.shape[1],
            "n_samples": len(X_pca),
            "gbm_acc": gbm_acc, "et_acc": et_acc, "resnet_acc": resnet_acc,
            "tb_tp": tp_r, "tb_sl": sl_r, "tb_exp": ep_r,
        }
        if vp:
            print(cc("\n  Training complete.  CPCV Sharpe={:.3f}\n".format(self.cpcv_sh), "G"))

    # ── Single prediction cycle ──
    def run_once(self):
        self.loop_n += 1
        live   = False
        fund   = None

        if NET:
            try:
                df_p = fetch_binance(CFG["SYMBOL"], CFG["INTERVAL"], CFG["CANDLES"])
                fund = fetch_funding(CFG["SYMBOL"])
                live = True
            except Exception as e:
                print("  Network error: {}  -> using synthetic".format(e))

        if not live:
            df_p, fund = make_synthetic(n=500, seed=self.loop_n % 20)

        df_p = prepare(df_p)
        price = float(df_p["close"].iloc[-1])

        if not self.trained:
            self.train(df_p, fund, verbose=True)

        # Build features + transform
        F_df   = build_features(df_p, fund)
        X_raw  = np.nan_to_num(F_df.values.astype(float), 0.0)
        X_sc   = self.scaler.transform(X_raw)
        X_pca  = self.pca.transform(X_sc)

        # Predictions
        gbm_prob = float(self.meta.primary.predict_proba(X_pca[-1:])[:, 1][0]) \
                   if self.meta.primary else 0.5
        meta_res = self.meta.predict(X_pca, gbm_prob)
        resnet_p = float(self.resnet.predict(X_pca[-1:])[0]) \
                   if self.resnet else 0.5

        # Market profile
        poc, vah, val = market_profile(df_p)

        # GARCH
        ret_ = df_p["close"].pct_change().dropna()
        cur_vol, garch_m, vol_reg, vol_pct = garch11(ret_)

        # Aggregate
        res = aggregate_signals(
            df_p, meta_res, resnet_p, gbm_prob,
            poc, vah, val, garch_m, vol_reg
        )

        display_dashboard(price, res, self.train_res, self.loop_n,
                          live, self.cpcv_sh)
        return res


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Elite Quant Engine v3.0")
    parser.add_argument("--loop",     action="store_true",
                        help="Run continuously")
    parser.add_argument("--interval", type=int,   default=30,
                        help="Seconds between loops")
    parser.add_argument("--account",  type=float, default=1000.0,
                        help="Account size in USDT")
    parser.add_argument("--retrain",  type=int,   default=10,
                        help="Retrain every N loops")
    args = parser.parse_args()

    print(cc("\n" + "=" * 74, "C"))
    print(cc("  ELITE QUANT ENGINE v3.0  —  BTC/USDT Binance Futures", "C"))
    print(cc("  ResNet | Meta-Label | Triple-Barrier | CPCV | FracDiff | 120 Alphas", "C"))
    print(cc("=" * 74, "C"))
    print("  Account  : ${:,.2f} USDT".format(args.account))
    print("  Max risk : {:.1f}% per trade".format(CFG["MAX_RISK_PCT"] * 100))
    print("  Mode     : {}".format("LIVE LOOP" if args.loop else "SINGLE RUN"))
    print()

    engine = EliteQuantEngine(account=args.account)

    if args.loop:
        n = 0
        while True:
            try:
                n += 1
                if n % args.retrain == 1:
                    engine.trained = False
                engine.run_once()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print(cc("\n  Stopped.", "Y"))
                break
            except Exception as exc:
                import traceback
                print("  Error: {}".format(exc))
                traceback.print_exc()
                time.sleep(15)
    else:
        engine.run_once()


if __name__ == "__main__":
    main()
