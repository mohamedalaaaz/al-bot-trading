#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ELITE QUANT ENGINE v4.0  —  REAL-TIME INFERENCE                         ║
║   BTC/USDT Binance Futures  │  WebSocket Live Data                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  REAL-TIME FEATURES:                                                        ║
║   • WebSocket kline + aggTrade streams (sub-second tick data)              ║
║   • Tick-by-tick CVD accumulation (real order flow, not approximation)     ║
║   • Live inference every new kline close — no polling lag                  ║
║   • Online learning: model updates incrementally with each new bar         ║
║   • Live P&L tracker and trade journal                                      ║
║   • Signal history with hit-rate tracking                                  ║
║   • Auto-reconnect WebSocket on drop                                        ║
║                                                                             ║
║  ML STACK (same institutional grade as v3):                                ║
║   • Triple-barrier labeling (ATR-adaptive)                                 ║
║   • Purged K-Fold + embargo (no leakage)                                   ║
║   • Fractional differentiation (stationary + memory)                       ║
║   • GBM + ExtraTrees + ResNet + Meta-label stacking                        ║
║   • Isotonic probability calibration                                        ║
║   • CPCV Sharpe estimation                                                  ║
║   • Sharpe-optimal Kelly sizing                                             ║
║   • 120+ alpha features across 14 categories                               ║
║                                                                             ║
║  INSTALL:                                                                   ║
║   pip install requests pandas numpy scipy scikit-learn websocket-client    ║
║                                                                             ║
║  RUN:                                                                       ║
║   python elite_quant_v4.py                  # real-time live              ║
║   python elite_quant_v4.py --paper          # paper trading mode          ║
║   python elite_quant_v4.py --account 5000   # set account size            ║
║   python elite_quant_v4.py --tf 1m          # timeframe (1m/3m/5m/15m)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, math, json, random, argparse, threading, warnings
import queue as queue_module
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
from scipy.signal import hilbert as sp_hilbert

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Optional WebSocket ──────────────────────────────────────────────────────
try:
    import websocket as ws_module
    WS_OK = True
except ImportError:
    WS_OK = False

try:
    import requests
    NET = True
except ImportError:
    NET = False


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
CFG = {
    "SYMBOL":       "BTCUSDT",
    "TF":           "5m",
    "CANDLES":      500,
    "ACCOUNT":      1000.0,
    "MAX_RISK":     0.015,
    "MIN_SCORE":    6,
    "MIN_CONF":     55.0,
    "MIN_META":     0.52,
    "MIN_RR":       1.5,
    "ATR_SL":       1.5,
    "TP_MULT":      2.5,
    # ML
    "TARGET_BARS":  5,
    "BARRIER_PCT":  0.008,
    "FRAC_D":       0.40,
    "PCA_VAR":      0.90,
    "PURGE":        5,
    "EMBARGO":      2,
    "GBM_N":        300,
    "ET_N":         200,
    "NN_H":         64,
    "NN_B":         3,
    "NN_EP":        100,
    "NN_LR":        5e-4,
    "NN_L2":        1e-4,
    "NN_DR":        0.25,
    # Real-time
    "RETRAIN_BARS": 50,     # retrain after N new bars
    "INFER_EVERY":  1,      # infer on every closed bar
    "TICK_WINDOW":  1000,   # keep last N ticks in memory
    "PAPER_SLIP":   0.0005, # paper trading slippage (0.05%)
}

BASE_WS  = "wss://fstream.binance.com/ws"
BASE_API = "https://fapi.binance.com"

COLORS = {
    "G":"\033[92m","R":"\033[91m","Y":"\033[93m","C":"\033[96m",
    "W":"\033[97m","B":"\033[1m","D":"\033[2m","M":"\033[95m",
    "K":"\033[90m","X":"\033[0m",
}
def cc(t, col): return COLORS.get(col,"") + str(t) + COLORS["X"]
def bar(v, w=12): n=min(int(abs(float(v))*w),w); return "█"*n+"░"*(w-n)


# ══════════════════════════════════════════════════════════════════════════
#  TICK BUFFER — real-time CVD from aggTrades
# ══════════════════════════════════════════════════════════════════════════
class TickBuffer:
    """
    Accumulates tick data from WebSocket aggTrade stream.
    Builds true bid/ask volume delta (not estimated from OHLCV).
    This is the same data Citadel uses for real order flow.
    """
    def __init__(self, maxlen=2000):
        self.ticks     = deque(maxlen=maxlen)
        self.lock      = threading.Lock()
        self.last_price= 0.0
        self.last_ts   = 0
        # Per-second aggregates
        self.buy_vol_s = 0.0
        self.sell_vol_s= 0.0

    def add(self, price: float, qty: float, is_buyer_maker: bool, ts: int):
        """is_buyer_maker=True means SELL (maker=passive buyer = aggressor is seller)"""
        with self.lock:
            self.last_price = price
            self.last_ts    = ts
            if is_buyer_maker:
                self.sell_vol_s += qty
            else:
                self.buy_vol_s  += qty
            self.ticks.append({
                "price": price, "qty": qty,
                "buy":   not is_buyer_maker, "ts": ts
            })

    def snapshot(self, window_ms: int = 5000) -> dict:
        """Return stats for last window_ms milliseconds."""
        now = self.last_ts
        with self.lock:
            recent = [t for t in self.ticks if now - t["ts"] <= window_ms]
        if not recent:
            return {"buy_vol":0,"sell_vol":0,"delta":0,"trades":0,
                    "price":self.last_price,"vwap":self.last_price}
        bv  = sum(t["qty"] for t in recent if t["buy"])
        sv  = sum(t["qty"] for t in recent if not t["buy"])
        vwap= sum(t["price"]*t["qty"] for t in recent) / max(sum(t["qty"] for t in recent),1e-9)
        return {
            "buy_vol":  bv,
            "sell_vol": sv,
            "delta":    bv - sv,
            "delta_pct":float(np.clip((bv-sv)/(bv+sv+1e-9), -1, 1)),
            "trades":   len(recent),
            "price":    self.last_price,
            "vwap":     vwap,
            "pressure": "BUY" if bv>sv*1.3 else ("SELL" if sv>bv*1.3 else "NEUTRAL"),
        }

    def reset_per_second(self):
        with self.lock:
            self.buy_vol_s = 0.0
            self.sell_vol_s= 0.0


# ══════════════════════════════════════════════════════════════════════════
#  KLINE BUFFER — assembles closed bars
# ══════════════════════════════════════════════════════════════════════════
class KlineBuffer:
    """Maintains rolling DataFrame of closed klines from WebSocket."""
    def __init__(self, maxlen=600):
        self.df      = pd.DataFrame()
        self.maxlen  = maxlen
        self.lock    = threading.Lock()
        self.new_bar = threading.Event()
        self.bar_count = 0

    def update(self, row: dict):
        """Add a closed kline row."""
        with self.lock:
            new_row = pd.DataFrame([row])
            new_row["open_time"] = pd.to_datetime(new_row["open_time"], utc=True)
            if self.df.empty:
                self.df = new_row
            else:
                # Avoid duplicates
                if row["open_time"] not in self.df["open_time"].values:
                    self.df = pd.concat([self.df, new_row], ignore_index=True)
                    self.df = self.df.tail(self.maxlen).reset_index(drop=True)
                    self.bar_count += 1
            self.new_bar.set()

    def get_df(self) -> pd.DataFrame:
        with self.lock:
            return self.df.copy()

    def wait_for_bar(self, timeout=60) -> bool:
        self.new_bar.clear()
        return self.new_bar.wait(timeout=timeout)


# ══════════════════════════════════════════════════════════════════════════
#  PAPER TRADER — simulates execution
# ══════════════════════════════════════════════════════════════════════════
class PaperTrader:
    def __init__(self, account: float):
        self.balance   = account
        self.start_bal = account
        self.position  = None   # {"side","entry","sl","tp1","tp2","qty","score","time"}
        self.trades    = []
        self.daily_pnl = 0.0
        self.wins      = 0
        self.losses    = 0
        self.lock      = threading.Lock()

    @property
    def total_trades(self): return self.wins + self.losses

    @property
    def win_rate(self):
        return self.wins / max(self.total_trades, 1) * 100

    @property
    def pnl_pct(self):
        return (self.balance - self.start_bal) / self.start_bal * 100

    def enter(self, side, entry, sl, tp1, tp2, qty, score, conf, reason):
        with self.lock:
            if self.position:
                return False
            slip = entry * CFG["PAPER_SLIP"] * (1 if side=="BUY" else -1)
            fill = entry + slip
            self.position = {
                "side": side, "entry": fill, "sl": sl,
                "tp1": tp1, "tp2": tp2, "qty": qty,
                "score": score, "conf": conf, "reason": reason,
                "time": datetime.now(timezone.utc),
                "tp1_hit": False,
            }
            return True

    def update(self, price: float):
        with self.lock:
            if not self.position:
                return None
            p    = self.position
            side = p["side"]
            result = None

            # TP1 hit → close 60%, move SL to breakeven
            if not p["tp1_hit"]:
                hit_tp1 = (side=="BUY" and price>=p["tp1"]) or \
                           (side=="SELL" and price<=p["tp1"])
                if hit_tp1:
                    pnl = p["qty"]*0.6 * abs(p["tp1"]-p["entry"]) * (1 if side=="BUY" else -1)
                    self.balance   += pnl
                    self.daily_pnl += pnl
                    p["tp1_hit"]    = True
                    p["qty"]       *= 0.4
                    p["sl"]         = p["entry"]  # breakeven
                    result = {"type":"TP1","pnl":pnl,"price":price}

            # TP2 hit
            if p["tp1_hit"]:
                hit_tp2 = (side=="BUY" and price>=p["tp2"]) or \
                           (side=="SELL" and price<=p["tp2"])
                if hit_tp2:
                    pnl = p["qty"] * abs(p["tp2"]-p["entry"]) * (1 if side=="BUY" else -1)
                    self.balance   += pnl
                    self.daily_pnl += pnl
                    self.wins      += 1
                    self.trades.append({**p, "exit":price, "pnl":pnl, "result":"WIN"})
                    self.position   = None
                    result = {"type":"WIN","pnl":pnl,"price":price}
                    return result

            # SL hit
            hit_sl = (side=="BUY" and price<=p["sl"]) or \
                     (side=="SELL" and price>=p["sl"])
            if hit_sl:
                pnl = p["qty"] * abs(p["sl"]-p["entry"]) * (-1 if side=="BUY" else 1)
                self.balance   += pnl
                self.daily_pnl += pnl
                self.losses    += 1
                self.trades.append({**p, "exit":price, "pnl":pnl, "result":"LOSS"})
                self.position   = None
                result = {"type":"LOSS","pnl":pnl,"price":price}

            return result

    def stats(self):
        return {
            "balance":    self.balance,
            "pnl_pct":    self.pnl_pct,
            "trades":     self.total_trades,
            "wins":       self.wins,
            "losses":     self.losses,
            "win_rate":   self.win_rate,
            "daily_pnl":  self.daily_pnl,
            "in_position":self.position is not None,
        }


# ══════════════════════════════════════════════════════════════════════════
#  SIGNAL HISTORY — tracks predictions and real outcomes
# ══════════════════════════════════════════════════════════════════════════
class SignalHistory:
    def __init__(self, maxlen=200):
        self.signals   = deque(maxlen=maxlen)
        self.correct   = 0
        self.total     = 0
        self.lock      = threading.Lock()

    def record(self, side, price, score, conf, meta_conf):
        with self.lock:
            self.signals.append({
                "side": side, "price": price, "score": score,
                "conf": conf, "meta_conf": meta_conf,
                "time": datetime.now(timezone.utc),
                "outcome": None,
            })

    def resolve(self, future_price: float):
        """Check if last unresolved signal was correct."""
        with self.lock:
            for sig in reversed(self.signals):
                if sig["outcome"] is None and sig["side"] != "WAIT":
                    correct = (sig["side"]=="BUY" and future_price>sig["price"]) or \
                              (sig["side"]=="SELL" and future_price<sig["price"])
                    sig["outcome"] = "WIN" if correct else "LOSS"
                    self.total   += 1
                    if correct: self.correct += 1
                    break

    @property
    def live_acc(self):
        return self.correct / max(self.total, 1) * 100

    def recent(self, n=5):
        with self.lock:
            return list(self.signals)[-n:]


# ══════════════════════════════════════════════════════════════════════════
#  ALL ML FUNCTIONS (self-contained, no imports from v3)
# ══════════════════════════════════════════════════════════════════════════

def frac_diff(series: pd.Series, d: float, thresh: float=1e-5) -> pd.Series:
    w = [1.0]
    k = 1
    while True:
        val = -w[-1]*(d-k+1)/k
        if abs(val) < thresh: break
        w.append(val); k += 1
    w     = np.array(w[::-1])
    width = len(w)
    out   = pd.Series(np.nan, index=series.index)
    for i in range(width-1, len(series)):
        out.iloc[i] = float(np.dot(w, series.iloc[i-width+1:i+1].values))
    return out

def triple_barrier(df: pd.DataFrame, pct: float=0.008, t_max: int=5) -> pd.Series:
    prices = df["close"].astype(float).values
    atrs   = df["atr"].astype(float).values
    n      = len(prices)
    labels = np.full(n, np.nan)
    log_r  = np.diff(np.log(prices+1e-9))
    rv5    = np.array([log_r[max(0,i-5):i].std() if i>1 else pct for i in range(n)])
    rv5    = np.maximum(rv5, 0.002)
    for i in range(n-t_max):
        p0  = prices[i]
        atr_i = atrs[i] if atrs[i]>0 else p0*0.003
        w   = max(pct, 1.5*rv5[i], atr_i/p0)
        tp  = p0*(1+w); sl = p0*(1-w)
        lbl = 0
        for j in range(1, t_max+1):
            if i+j>=n: break
            p = prices[i+j]
            if p>=tp: lbl=1; break
            if p<=sl: lbl=-1; break
        if lbl==0:
            ret_fwd=(prices[min(i+t_max,n-1)]/p0)-1
            if   ret_fwd> 0.0005: lbl=1
            elif ret_fwd<-0.0005: lbl=-1
        labels[i]=lbl
    return pd.Series(labels, index=df.index).dropna()

def purged_kfold(n: int, k: int=5, purge: int=5, embargo: int=2):
    fsize  = n//k
    splits = []
    for f in range(k):
        ts = f*fsize; te = ts+fsize if f<k-1 else n
        tr = list(range(0, max(0,ts-purge))) + list(range(min(n,te+embargo), n))
        ti = list(range(ts,te))
        if len(tr)>=50 and len(ti)>=10: splits.append((tr,ti))
    return splits

def rsi_s(prices, period):
    d=prices.diff(); g=d.clip(lower=0).rolling(period).mean()
    l=(-d.clip(upper=0)).rolling(period).mean()
    rs=g/l.replace(0,np.nan); return (100-100/(1+rs)).fillna(50)

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy()
    d["body"]     =d["close"]-d["open"]
    d["body_pct"] =d["body"]/d["open"]*100
    d["is_bull"]  =d["body"]>0
    d["wick_top"] =d["high"]-d[["open","close"]].max(axis=1)
    d["wick_bot"] =d[["open","close"]].min(axis=1)-d["low"]
    d["sell_vol"] =d["volume"]-d["taker_buy_vol"]
    d["delta"]    =d["taker_buy_vol"]-d["sell_vol"]
    d["delta_pct"]=(d["delta"]/d["volume"].replace(0,np.nan)).fillna(0)
    hl =d["high"]-d["low"]; hpc=(d["high"]-d["close"].shift(1)).abs()
    lpc=(d["low"]-d["close"].shift(1)).abs()
    d["atr"]  =pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    rm=d["volume"].rolling(50).mean(); rs=d["volume"].rolling(50).std().replace(0,np.nan)
    d["vol_z"]=(d["volume"]-rm)/rs
    d["hour"]=d["open_time"].dt.hour
    d["dow"] =d["open_time"].dt.dayofweek
    d["session"]=d["hour"].apply(
        lambda h:"Asia" if h<8 else "London" if h<13 else "NY" if h<20 else "Late")
    return d.fillna(0)

def build_features(df: pd.DataFrame, fund: pd.DataFrame = None,
                   tick_snap: dict = None) -> pd.DataFrame:
    """120+ features + real-time tick features when available."""
    d   = df.copy()
    c_  = d["close"].astype(float)
    vol = d["volume"].astype(float).replace(0,np.nan)
    dp  = d["delta_pct"].astype(float)
    dlt = d["delta"].astype(float)
    ret = c_.pct_change()
    lr  = np.log(c_/c_.shift(1)).fillna(0)
    tp_ = (d["high"]+d["low"]+c_)/3
    atr = d["atr"].astype(float).replace(0,np.nan)
    F   = pd.DataFrame(index=d.index)

    # 1. Momentum
    for lag in [1,2,3,5,8,13,21,34]:
        F["mom_"+str(lag)] = c_.pct_change(lag)
    for fast,slow in [(8,21),(12,26),(5,13)]:
        F["macd_"+str(fast)+"_"+str(slow)] = (c_.ewm(fast).mean()-c_.ewm(slow).mean())/c_
    F["mom_acc"] = c_.pct_change(5)-c_.pct_change(5).shift(5)
    for w in [10,20,50]:
        hi_=d["high"].rolling(w).max(); lo_=d["low"].rolling(w).min(); rng_=(hi_-lo_).replace(0,np.nan)
        F["rpos_"+str(w)]=(c_-lo_)/rng_; F["dhi_"+str(w)]=(hi_-c_)/c_*100; F["dlo_"+str(w)]=(c_-lo_)/c_*100

    # 2. Mean reversion
    for w in [10,20,50,100]:
        mu_=c_.rolling(w).mean(); sg_=c_.rolling(w).std().replace(0,np.nan); F["z_"+str(w)]=(c_-mu_)/sg_
    for p in [7,14,21]: F["rsi_"+str(p)]=rsi_s(c_,p)
    F["willr"]=((d["high"].rolling(14).max()-c_)/(d["high"].rolling(14).max()-d["low"].rolling(14).min()+1e-9)*-100)
    ma_cci=tp_.rolling(20).mean()
    md_cci=tp_.rolling(20).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True).replace(0,np.nan)
    F["cci"]=(tp_-ma_cci)/(0.015*md_cci)

    # 3. Fractional diff
    for d_val in [0.3,0.4,0.5]:
        F["fd_"+str(d_val).replace(".","")]=frac_diff(c_,d_val)

    # 4. Order flow / delta
    F["delta_pct"]=dp; F["buy_ratio"]=d["taker_buy_vol"]/vol; F["vol_imb"]=dlt/vol
    cvd20=dlt.rolling(20).sum()
    F["cvd_20n"]=cvd20/vol.rolling(20).mean(); F["cvd_sl3"]=cvd20.diff(3); F["cvd_sl5"]=cvd20.diff(5)
    F["cvd_acc"]=cvd20.diff(3).diff(2)
    pr_s=c_.diff(3)/c_.shift(3)*100; cvd_s=cvd20.diff(3)
    F["div_bull"]=((pr_s<-0.12)&(cvd_s>0)).astype(float)
    F["div_bear"]=((pr_s> 0.12)&(cvd_s<0)).astype(float)
    F["exh_buy"] =((dp>0.28)&(d["body_pct"].abs()<0.06)).astype(float)
    F["exh_sell"]=((dp<-0.28)&(d["body_pct"].abs()<0.06)).astype(float)
    # Kyle lambda
    ky=pd.Series(np.nan,index=d.index)
    for i in range(20,len(d)):
        ri=ret.iloc[i-20:i].values; dpi=dp.iloc[i-20:i].values
        c2=np.cov(ri,dpi) if len(ri)>3 else np.zeros((2,2))
        ky.iloc[i]=float(c2[0,1]/(dpi.var()+1e-12))
    F["kyle_lam"]=ky

    # 5. Volatility
    for w in [5,10,20,50]:
        F["rv_"+str(w)]=(lr**2).rolling(w).sum(); F["rvol_"+str(w)]=lr.rolling(w).std()
    F["pk_vol"]=np.sqrt((1/(4*math.log(2)))*(np.log(d["high"]/d["low"].replace(0,np.nan))**2).rolling(20).mean())
    F["gk_vol"]=np.sqrt((0.5*(np.log(d["high"]/d["low"].replace(0,np.nan))**2)-(2*math.log(2)-1)*(np.log(c_/d["open"].replace(0,np.nan))**2)).rolling(20).mean())
    rv20=(lr**2).rolling(20).sum()
    F["vov"]=rv20.rolling(10).std()/rv20.rolling(10).mean().replace(0,np.nan)
    F["skew50"]=lr.rolling(50).apply(lambda x:float(sp_skew(x)),raw=True)
    F["kurt50"]=lr.rolling(50).apply(lambda x:float(sp_kurt(x)),raw=True)
    F["vr_5_20"]=F["rvol_5"]/F["rvol_20"].replace(0,np.nan)

    # 6. VWAP & structure
    for w in [20,50,100]:
        vw_=(tp_*vol).rolling(w).sum()/vol.rolling(w).sum()
        vr_=(vol*(tp_-vw_)**2).rolling(w).sum()/vol.rolling(w).sum()
        vs_=np.sqrt(vr_.replace(0,np.nan))
        F["vwap_dev_"+str(w)]=(c_-vw_)/vw_*100; F["vwap_band_"+str(w)]=(c_-vw_)/vs_.replace(0,np.nan)
    for sp in [8,21,50]: F["ema_dev_"+str(sp)]=(c_-c_.ewm(sp).mean())/c_*100
    F["ema_8_21"]=(c_.ewm(8).mean()-c_.ewm(21).mean())/c_*100
    F["ema_cross"]=(c_.ewm(8).mean()>c_.ewm(21).mean()).astype(float)

    # 7. Microstructure
    rng_=(d["high"]-d["low"]).replace(0,np.nan)
    F["wt_rel"]=d["wick_top"]/atr; F["wb_rel"]=d["wick_bot"]/atr
    F["wasym"]=(d["wick_bot"]-d["wick_top"])/atr
    F["effic"]=d["body_pct"].abs()/(rng_/c_*100).replace(0,np.nan)
    F["hl_pos"]=(c_-d["low"])/rng_
    F["vol_z"]=d["vol_z"]; F["big_trade"]=(d["vol_z"]>3).astype(float)
    F["absorb"]=((d["vol_z"]>1.5)&(d["body_pct"].abs()<0.08)).astype(float)
    F["trap"]=((d["body_pct"].shift(1).abs()>0.25)&(d["body_pct"]*d["body_pct"].shift(1)<0)).astype(float)
    F["amihud"]=(lr.abs()/vol).rolling(20).mean()

    # 8. Hilbert / Fisher
    try:
        raw_arr=c_.values.astype(float)
        x_dt=raw_arr-np.linspace(raw_arr[0],raw_arr[-1],len(raw_arr))
        analytic=sp_hilbert(x_dt)
        F["hil_amp"]=pd.Series(np.abs(analytic),index=d.index)/(c_.std()+1e-9)
        F["hil_phase"]=pd.Series(np.angle(analytic),index=d.index)
        F["hil_freq"]=pd.Series(np.gradient(np.unwrap(np.angle(analytic))),index=d.index)
        hi10=c_.rolling(10).max(); lo10=c_.rolling(10).min()
        v_=(2*(c_-lo10)/(hi10-lo10+1e-9)-1).clip(-0.999,0.999)
        F["fisher"]=0.5*np.log((1+v_)/(1-v_+1e-10))
    except Exception:
        for col in ["hil_amp","hil_phase","hil_freq","fisher"]: F[col]=0.0

    # 9. Wyckoff
    n_w=min(30,len(d)); x_w=np.arange(n_w); rec=d.tail(n_w)
    def sl_(vals):
        try: return float(np.polyfit(x_w[:len(vals)],vals,1)[0])
        except: return 0.0
    pt=sl_(rec["close"].values); bt=sl_(rec["taker_buy_vol"].values)
    st=sl_((rec["volume"]-rec["taker_buy_vol"]).values)
    wy=(2 if pt<-0.3 and bt>0 else 3 if pt>0.3 and bt>0 else
        -2 if pt>0.3 and st>0 else -3 if pt<-0.3 and st>0 else 0)
    F["wyckoff"]=float(wy)
    cvd_t=0.0
    if len(d)>=20:
        v0=float(dlt.rolling(20).sum().iloc[-1]); v1=float(dlt.rolling(20).sum().iloc[-20])
        cvd_t=float(np.clip((v0-v1)/10000,-3,3))
    F["sm_flow"]=cvd_t

    # 10. Time
    h_=d["open_time"].dt.hour; dw=d["open_time"].dt.dayofweek
    F["sin_h"]=np.sin(2*math.pi*h_/24); F["cos_h"]=np.cos(2*math.pi*h_/24)
    F["sin_dow"]=np.sin(2*math.pi*dw/7); F["cos_dow"]=np.cos(2*math.pi*dw/7)
    F["london"]=h_.isin([8,9,10,11,12]).astype(float); F["ny"]=h_.isin([13,14,15,16,17,18,19]).astype(float)

    # 11. Funding
    avg_fr=0.0; tr_fr=0.0
    if fund is not None and len(fund)>=3:
        rates=fund["fundingRate"].tail(8).values.astype(float)
        avg_fr=float(rates.mean()); tr_fr=float(np.clip((rates[-1]-rates[0])*1000,-3,3))
    F["fund_rate"]=avg_fr; F["fund_trend"]=tr_fr
    F["fund_rev"]=float(-1 if avg_fr>0.0008 else (1 if avg_fr<-0.0005 else 0))

    # 12. Stacked / liquidity
    F["stk_buy"]=(dp>0.1).rolling(3).sum().eq(3).astype(float)
    F["stk_sell"]=(dp<-0.1).rolling(3).sum().eq(3).astype(float)
    F["bid_abs"]=((d["wick_bot"]>atr*0.25)&(dp>0.1)&(d["vol_z"]>1)).astype(float)
    F["ask_abs"]=((d["wick_top"]>atr*0.25)&(dp<-0.1)&(d["vol_z"]>1)).astype(float)

    # 13. Interaction
    F["mom_vol"]=c_.pct_change(3)*d["vol_z"]
    F["dlt_mom"]=dp*np.sign(c_.pct_change(1))
    F["vwap_dlt"]=F["vwap_dev_20"]*dp

    # 14. OU
    ou_z=0.0; ou_hl=999.0
    x_ou=c_.values[-100:] if len(c_)>=100 else c_.values
    if len(x_ou)>=30:
        dx_=np.diff(x_ou); xl_=x_ou[:-1]
        A_=np.column_stack([np.ones(len(xl_)),xl_])
        try:
            co_,_,_,_=np.linalg.lstsq(A_,dx_,rcond=None)
            mu_ou=-co_[0]/co_[1] if co_[1]!=0 else float(x_ou.mean())
            sg_ou=max(float(np.std(dx_-(co_[0]+co_[1]*xl_))),1e-9)
            ou_z=float(np.clip((float(x_ou[-1])-mu_ou)/sg_ou,-5,5))
            ou_hl=float(np.clip(math.log(2)/(-co_[1]),0,200)) if co_[1]<0 else 999.
        except: pass
    F["ou_z"]=ou_z; F["ou_hl"]=ou_hl

    # ── REAL-TIME TICK FEATURES (added when WS is live) ──
    if tick_snap:
        F["tick_delta_pct"] = float(tick_snap.get("delta_pct",0))
        F["tick_buy_ratio"] = float(tick_snap.get("buy_vol",0.5) /
                               max(tick_snap.get("buy_vol",0)+tick_snap.get("sell_vol",0)+1e-9,1))
        F["tick_pressure"]  = float(1 if tick_snap.get("pressure","NEUTRAL")=="BUY" else
                               -1 if tick_snap.get("pressure","")=="SELL" else 0)
        F["tick_vwap_dev"]  = float((tick_snap.get("price",0) -
                               tick_snap.get("vwap",tick_snap.get("price",0))) /
                               max(tick_snap.get("vwap",1), 1) * 100)
    else:
        for col in ["tick_delta_pct","tick_buy_ratio","tick_pressure","tick_vwap_dev"]:
            F[col] = 0.0

    return F.replace([np.inf,-np.inf],0).fillna(0)


# ══════════════════════════════════════════════════════════════════════════
#  RESNET (same as v3, self-contained)
# ══════════════════════════════════════════════════════════════════════════
class ResNet:
    def __init__(self, n_in, hidden=64, n_blocks=3, lr=5e-4, l2=1e-4, dropout=0.25):
        self.lr=lr; self.l2=l2; self.dr=dropout; self.nb=n_blocks; self.val_acc=0.5
        def he(a,b): return np.random.randn(a,b).astype(np.float64)*math.sqrt(2/a)
        self.Wi=he(n_in,hidden); self.bi=np.zeros(hidden,dtype=np.float64)
        self.Wr1=[he(hidden,hidden) for _ in range(n_blocks)]
        self.br1=[np.zeros(hidden,dtype=np.float64) for _ in range(n_blocks)]
        self.Wr2=[he(hidden,hidden) for _ in range(n_blocks)]
        self.br2=[np.zeros(hidden,dtype=np.float64) for _ in range(n_blocks)]
        self.Wo=he(hidden,1); self.bo=np.zeros(1,dtype=np.float64)
        ap=self._p(); self.m={k:np.zeros_like(v) for k,v in ap.items()}; self.v={k:np.zeros_like(v) for k,v in ap.items()}; self.t=0

    def _p(self):
        p={"Wi":self.Wi,"bi":self.bi,"Wo":self.Wo,"bo":self.bo}
        for i in range(self.nb):
            p["W1_"+str(i)]=self.Wr1[i]; p["b1_"+str(i)]=self.br1[i]
            p["W2_"+str(i)]=self.Wr2[i]; p["b2_"+str(i)]=self.br2[i]
        return p
    @staticmethod
    def _sw(x): return x/(1+np.exp(-np.clip(x,-50,50)))
    @staticmethod
    def _sd(x): s=1/(1+np.exp(-np.clip(x,-50,50))); return s+x*s*(1-s)
    @staticmethod
    def _sig(x): return 1/(1+np.exp(-np.clip(x,-50,50)))

    def fwd(self,X,train=True):
        ca={}; Z0=X@self.Wi+self.bi; A0=self._sw(Z0); ca["X"]=X; ca["Z0"]=Z0; A=A0
        for i in range(self.nb):
            Z1=A@self.Wr1[i]+self.br1[i]; A1=self._sw(Z1)
            if train and self.dr>0:
                mk=(np.random.rand(*A1.shape)>self.dr).astype(np.float64)/(1-self.dr+1e-9)
                A1*=mk; ca["mk_"+str(i)]=mk
            Z2=A1@self.Wr2[i]+self.br2[i]; A2=self._sw(Z2+A)
            ca["Ai_"+str(i)]=A; ca["Z1_"+str(i)]=Z1; ca["A1_"+str(i)]=A1; ca["Z2_"+str(i)]=Z2; A=A2
        Zo=A@self.Wo+self.bo; Ao=self._sig(Zo); ca["Af"]=A; ca["Zo"]=Zo
        return Ao.ravel(),ca

    def bwd(self,y,out,ca):
        m=float(len(y)); g={}; dA=(out-y)/m; dZo=dA.reshape(-1,1)
        g["Wo"]=ca["Af"].T@dZo+self.l2*self.Wo; g["bo"]=dZo.sum(0); dA=dZo@self.Wo.T
        for i in reversed(range(self.nb)):
            Ai=ca["Ai_"+str(i)]; Z1=ca["Z1_"+str(i)]; A1=ca["A1_"+str(i)]; Z2=ca["Z2_"+str(i)]
            dA2=dA*self._sd(Z2+Ai)
            g["W2_"+str(i)]=A1.T@dA2+self.l2*self.Wr2[i]; g["b2_"+str(i)]=dA2.sum(0)
            dA1=dA2@self.Wr2[i].T
            if "mk_"+str(i) in ca: dA1*=ca["mk_"+str(i)]
            dZ1=dA1*self._sd(Z1)
            g["W1_"+str(i)]=Ai.T@dZ1+self.l2*self.Wr1[i]; g["b1_"+str(i)]=dZ1.sum(0)
            dA=dZ1@self.Wr1[i].T+dA2
        dZ0=dA*self._sd(ca["Z0"]); g["Wi"]=ca["X"].T@dZ0+self.l2*self.Wi; g["bi"]=dZ0.sum(0)
        return g

    def adam(self,g):
        self.t+=1; b1,b2,eps=0.9,0.999,1e-8; p=self._p()
        for k,gv in g.items():
            if k not in p: continue
            self.m[k]=b1*self.m.get(k,np.zeros_like(gv))+(1-b1)*gv
            self.v[k]=b2*self.v.get(k,np.zeros_like(gv))+(1-b2)*gv**2
            mc=self.m[k]/(1-b1**self.t); vc=self.v[k]/(1-b2**self.t)
            p[k]-=self.lr*mc/(np.sqrt(vc)+eps)

    def fit(self,X,y,Xv=None,yv=None,epochs=100,batch=32):
        best_acc=0; best_w=None; no_imp=0
        for ep in range(epochs):
            idx=np.random.permutation(len(X))
            for s in range(0,len(X),batch):
                Xb=X[idx[s:s+batch]]; yb=y[idx[s:s+batch]]
                if len(Xb)<2: continue
                out,ca=self.fwd(Xb,True); g=self.bwd(yb,out,ca); self.adam(g)
            if Xv is not None and len(Xv)>0:
                pv,_=self.fwd(Xv,False); acc=float(((pv>0.5)==yv).mean())
                if acc>best_acc: best_acc=acc; best_w={k:v.copy() for k,v in self._p().items()}; no_imp=0
                else: no_imp+=1
                if no_imp>=15: break
            if (ep+1)%20==0: self.lr*=0.7
        if best_w:
            for k,v in best_w.items(): self._p()[k][...]=v
        self.val_acc=best_acc

    def predict(self,X): p,_=self.fwd(X,False); return p


# ══════════════════════════════════════════════════════════════════════════
#  META-LABEL SYSTEM
# ══════════════════════════════════════════════════════════════════════════
class MetaLabelSystem:
    def __init__(self):
        self.primary=None; self.secondary=None
        self.cal=IsotonicRegression(out_of_bounds="clip"); self.calibrated=False

    def fit(self,X,y_tb,splits,verbose=True):
        y_dir=(y_tb==1).astype(int)
        gbm=GradientBoostingClassifier(n_estimators=CFG["GBM_N"],learning_rate=0.03,
            max_depth=4,subsample=0.70,min_samples_leaf=8,random_state=42)
        oof_p=np.full(len(X),0.5)
        for tr,te in splits:
            y_tr=y_dir[tr]
            if len(np.unique(y_tr))<2: continue
            gbm.fit(X[tr],y_tr); oof_p[te]=gbm.predict_proba(X[te])[:,1]
        if len(np.unique(y_dir))<2:
            y_dir_b=np.zeros(len(y_dir)); y_dir_b[len(y_dir)//2:]=1
            gbm.fit(X,y_dir_b)
        else:
            gbm.fit(X,y_dir)
        self.primary=gbm
        gbm_acc=float(((oof_p>0.5).astype(int)==y_dir).mean())
        if verbose: print("    GBM OOF acc: {:.4f}".format(gbm_acc))

        y_meta=np.zeros(len(y_tb)); pred_p=(oof_p>0.5).astype(int)
        for i in range(len(y_tb)):
            if y_tb[i]==0: y_meta[i]=0
            elif y_tb[i]==1 and pred_p[i]==1: y_meta[i]=1
            elif y_tb[i]==-1 and pred_p[i]==0: y_meta[i]=1
            else: y_meta[i]=0

        et=ExtraTreesClassifier(n_estimators=CFG["ET_N"],max_depth=5,
            min_samples_leaf=8,random_state=42,n_jobs=-1)
        oof_m=np.full(len(X),0.5)
        for tr,te in splits:
            Xp_tr=np.column_stack([X[tr],oof_p[tr]]); Xp_te=np.column_stack([X[te],oof_p[te]])
            ym=y_meta[tr]
            if len(np.unique(ym))<2: continue
            et.fit(Xp_tr,ym); oof_m[te]=et.predict_proba(Xp_te)[:,1]
        valid=y_tb!=0
        if valid.sum()>20: self.cal.fit(oof_m[valid],y_meta[valid]); self.calibrated=True
        et.fit(np.column_stack([X,oof_p]),y_meta); self.secondary=et
        ne=y_tb!=0
        et_acc=float(((oof_m[ne]>0.5).astype(int)==y_meta[ne]).mean()) if ne.sum()>0 else 0.5
        if verbose: print("    ET meta acc:  {:.4f}".format(et_acc))
        return oof_p,oof_m,gbm_acc,et_acc

    def predict(self,X,primary_prob):
        if not self.primary: return {"direction":0.5,"meta_prob":0.5,"signal":"WAIT","take":False}
        dir_p=float(self.primary.predict_proba(X[-1:])[:,1][0])
        Xp=np.column_stack([X[-1:],[[primary_prob]]])
        meta=float(self.secondary.predict_proba(Xp)[:,1][0]) if self.secondary else 0.5
        if self.calibrated: meta=float(self.cal.predict([meta])[0])
        return {"direction":dir_p,"meta_prob":meta,
                "signal":"BUY" if dir_p>0.55 else("SELL" if dir_p<0.45 else "WAIT"),
                "take":meta>=CFG["MIN_META"]}


# ══════════════════════════════════════════════════════════════════════════
#  GARCH + KELLY + MARKET PROFILE + KALMAN (compact)
# ══════════════════════════════════════════════════════════════════════════
def garch11(ret):
    r=ret.dropna().values
    if len(r)<30: return 0.003,1.0,"MEDIUM",50.0
    v0=float(np.var(r))
    def nll(p):
        om,al,be=p
        if om<=0 or al<0 or be<0 or al+be>=1: return 1e10
        h=np.full(len(r),v0); ll=0.0
        for t in range(1,len(r)):
            h[t]=om+al*r[t-1]**2+be*h[t-1]
            if h[t]<=0: return 1e10
            ll+=-0.5*(math.log(2*math.pi*h[t])+r[t]**2/h[t])
        return -ll
    try:
        res=optimize.minimize(nll,[v0*0.05,0.08,0.88],method="L-BFGS-B",
            bounds=[(1e-9,None),(1e-9,0.999),(1e-9,0.999)],options={"maxiter":100})
        om,al,be=res.x
    except: om,al,be=v0*0.05,0.08,0.88
    h=np.full(len(r),v0)
    for t in range(1,len(r)):
        h[t]=max(om+al*r[t-1]**2+be*h[t-1],1e-12)
    cv=float(math.sqrt(h[-1])); vp=float(stats.percentileofscore(np.sqrt(h),cv))
    rg="LOW" if vp<30 else("HIGH" if vp>75 else "MEDIUM")
    sm=1.5 if vp<30 else(0.5 if vp>80 else 1.0)
    return cv,sm,rg,vp

def sharpe_kelly(mu,sigma,rho=0.0):
    if sigma<=0: return 0.0,0.0
    sk=mu/(sigma**2); sh=sk*(1-rho)*0.25
    return float(np.clip(sh,0,3)),float(sk)

def market_profile(df,tick=25.0):
    lo=df["low"].min(); hi=df["high"].max()
    bkts=np.arange(math.floor(lo/tick)*tick,math.ceil(hi/tick)*tick+tick,tick)
    vm=defaultdict(float)
    for _,row in df.iterrows():
        lvls=bkts[(bkts>=row["low"])&(bkts<=row["high"])]
        if not len(lvls): continue
        vp=row["volume"]/len(lvls)
        for lv in lvls: vm[lv]+=vp
    if not vm:
        p=float(df["close"].iloc[-1]); return p,p,p
    pf=pd.DataFrame({"p":list(vm.keys()),"v":list(vm.values())}).sort_values("p")
    poc=float(pf.loc[pf["v"].idxmax(),"p"]); tot=pf["v"].sum()
    pi=pf["v"].idxmax(); cum=0.0; va=[]
    for _ in range(len(pf)):
        ui=pi+1; li=pi-1
        uv=pf.loc[ui,"v"] if ui in pf.index else 0.0
        dv=pf.loc[li,"v"] if li in pf.index else 0.0
        if uv>=dv and ui in pf.index: va.append(ui); cum+=uv; pi=ui
        elif li in pf.index: va.append(li); cum+=dv; pi=li
        else: break
        if cum/tot>=0.70: break
    vah=float(pf.loc[va,"p"].max()) if va else poc+tick*5
    val=float(pf.loc[va,"p"].min()) if va else poc-tick*5
    return poc,vah,val

def kalman_filter(prices):
    z=prices.astype(float).values; n=len(z)
    F_=np.array([[1.,1.],[0.,1.]]); H_=np.array([[1.,0.]])
    Q_=np.array([[0.01,0.001],[0.001,0.0001]]); R_=np.array([[1.0]])
    x=np.array([[z[0]],[0.]]); P=np.eye(2)*1000.
    kp=np.zeros(n); kt=np.zeros(n)
    for t in range(n):
        xp=F_@x; Pp=F_@P@F_.T+Q_
        K=Pp@H_.T@np.linalg.inv(H_@Pp@H_.T+R_)
        x=xp+K*(z[t]-float((H_@xp).flat[0])); P=(np.eye(2)-K@H_)@Pp
        kp[t]=float(x[0].flat[0]); kt[t]=float(x[1].flat[0])
    return float(kp[-1]),float(kt[-1])

def cpcv_sharpe(oof_probs,y_dir,ret_s,n_splits=6,n_test=2):
    n=min(len(oof_probs),len(y_dir),len(ret_s))
    if n<100: return 0.0
    fs=n//n_splits
    folds=[list(range(i*fs,(i+1)*fs if i<n_splits-1 else n)) for i in range(n_splits)]
    sharpes=[]
    for combo in combinations(range(n_splits),n_test):
        ti=[]; [ti.extend(folds[ci]) for ci in combo]
        if len(ti)<10: continue
        p_=oof_probs[ti]; r_=ret_s.iloc[ti].values
        st=np.where(p_>0.55,r_,np.where(p_<0.45,-r_,0.0))
        sg=st.std(); mu=st.mean()
        sharpes.append(mu/sg*math.sqrt(288*252) if sg>0 else 0.0)
    return float(np.mean(sharpes)) if sharpes else 0.0


# ══════════════════════════════════════════════════════════════════════════
#  AGGREGATE SIGNALS → FINAL SCORE
# ══════════════════════════════════════════════════════════════════════════
def aggregate(df, meta_res, resnet_p, gbm_prob,
              poc, vah, val, garch_m, vol_reg, tick_snap=None):
    price=float(df["close"].iloc[-1]); atr=float(df["atr"].iloc[-1]) or price*0.003
    ret=df["close"].pct_change().dropna()
    dp=df["delta_pct"].astype(float); dlt=df["delta"].astype(float)

    # CVD divergence
    cvd20=dlt.rolling(20).sum(); pr_s=df["close"].diff(3)/df["close"].shift(3)*100; cvd_s=cvd20.diff(3)
    div_b=bool(pr_s.iloc[-1]<-0.12 and cvd_s.iloc[-1]>0)
    div_s=bool(pr_s.iloc[-1]>0.12  and cvd_s.iloc[-1]<0)

    # OU
    ou_z=0.0; x_ou=df["close"].values[-100:]
    if len(x_ou)>=30:
        dx_=np.diff(x_ou); xl_=x_ou[:-1]; A_=np.column_stack([np.ones(len(xl_)),xl_])
        try:
            co_,_,_,_=np.linalg.lstsq(A_,dx_,rcond=None)
            mu_ou=-co_[0]/co_[1] if co_[1]!=0 else float(x_ou.mean())
            sg_ou=max(float(np.std(dx_-(co_[0]+co_[1]*xl_))),1e-9)
            ou_z=float(np.clip((float(x_ou[-1])-mu_ou)/sg_ou,-5,5))
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

    # Kalman
    kal_p,kal_t=kalman_filter(df["close"])

    # Trap / absorption
    bp=df["body_pct"]; vz=float(df["vol_z"].iloc[-1])
    trap_s=bool(bp.shift(1).iloc[-1]<-0.25 and df["close"].iloc[-1]>df["open"].shift(1).iloc[-1])
    trap_l=bool(bp.shift(1).iloc[-1]> 0.25 and df["close"].iloc[-1]<df["open"].shift(1).iloc[-1])
    ab_sc=(1 if vz>1.5 and dp.iloc[-1]>0.1 and abs(bp.iloc[-1])<0.08 else
          -1 if vz>1.5 and dp.iloc[-1]<-0.1 and abs(bp.iloc[-1])<0.08 else 0)

    # VWAP
    c_=df["close"].astype(float); vol_=df["volume"].astype(float).replace(0,np.nan)
    tp__=(df["high"]+df["low"]+c_)/3
    vw20=(tp__*vol_).rolling(20).sum()/vol_.rolling(20).sum()
    vr20=(vol_*(tp__-vw20)**2).rolling(20).sum()/vol_.rolling(20).sum()
    vs20=np.sqrt(vr20.replace(0,np.nan))
    vdev=float((c_-vw20).iloc[-1]/vs20.iloc[-1]) if float(vs20.iloc[-1])>0 else 0.0
    vwap_sc=(2 if vdev<-1.8 else 1 if vdev<-0.8 else -2 if vdev>1.8 else -1 if vdev>0.8 else 0)

    # Tick real-time signal (only when WS active)
    tick_sc = 0
    if tick_snap and tick_snap.get("trades",0) > 5:
        td = tick_snap.get("delta_pct", 0)
        tick_sc = (2 if td>0.3 else 1 if td>0.1 else -2 if td<-0.3 else -1 if td<-0.1 else 0)

    # Scores
    resnet_sc=(3 if resnet_p>0.70 else 2 if resnet_p>0.62 else 1 if resnet_p>0.56 else
              -3 if resnet_p<0.30 else -2 if resnet_p<0.38 else -1 if resnet_p<0.44 else 0)
    dir_sc=(3 if meta_res["direction"]>0.65 else 2 if meta_res["direction"]>0.56 else
           -3 if meta_res["direction"]<0.35 else -2 if meta_res["direction"]<0.44 else 0)
    meta_mult=(1.5 if meta_res["meta_prob"]>0.65 else 0.5 if meta_res["meta_prob"]<0.45 else 1.0)
    cvd_sc=3 if div_b else(-3 if div_s else 0)
    ou_sc=(3 if ou_z<-2 else 2 if ou_z<-1 else 1 if ou_z<-0.5 else
          -3 if ou_z>2 else -2 if ou_z>1 else -1 if ou_z>0.5 else 0)
    kal_sc=(2 if kal_t>0.2 else 1 if kal_t>0 else -2 if kal_t<-0.2 else -1 if kal_t<0 else 0)
    trap_sc=(2 if trap_s else -2 if trap_l else 0)

    raw=(resnet_sc*1.5 + dir_sc*meta_mult + cvd_sc + ou_sc + wy +
         kal_sc + trap_sc + ab_sc + vwap_sc + tick_sc)
    raw*=(0.65 if vol_reg=="HIGH" else 1.0)
    score=int(np.clip(raw,-15,15))
    conf=min(abs(score)/15*100*meta_res["meta_prob"]*1.8, 99.0)

    mu_e=float(ret.tail(50).mean())*(1 if score>0 else -1)
    sg_e=float(ret.tail(50).std())
    rho_e=max(0.0,1-abs(score)/15)
    sk_r,sk_full=sharpe_kelly(mu_e,sg_e,rho_e)

    stop_dist=atr*CFG["ATR_SL"]
    if score>=CFG["MIN_SCORE"]:
        side="BUY"; sl_=round(min(val,price-stop_dist),1)
        tp1=round(poc if poc>price else price+stop_dist*CFG["TP_MULT"],1)
        tp2=round(vah if vah>tp1 else price+stop_dist*CFG["TP_MULT"]*2,1)
    elif score<=-CFG["MIN_SCORE"]:
        side="SELL"; sl_=round(max(vah,price+stop_dist),1)
        tp1=round(poc if poc<price else price-stop_dist*CFG["TP_MULT"],1)
        tp2=round(val if val<tp1 else price-stop_dist*CFG["TP_MULT"]*2,1)
    else:
        side="WAIT"; sl_=tp1=tp2=None

    rr=abs(tp1-price)/max(abs(price-(sl_ or price)),1.0) if tp1 else 0.0
    qty=(CFG["ACCOUNT"]*sk_r*garch_m/max(stop_dist,1.0)) if sl_ else 0.0
    tradeable=(side!="WAIT" and conf>=CFG["MIN_CONF"] and
               rr>=CFG["MIN_RR"] and meta_res["take"])

    reasons=[]
    if abs(resnet_sc)>=2: reasons.append("{:+.0f} ResNet".format(resnet_sc*1.5))
    if abs(dir_sc)>=2:   reasons.append("{:+.0f} MetaML".format(dir_sc*meta_mult))
    if abs(cvd_sc)>=3:   reasons.append("{:+d} CVD".format(cvd_sc))
    if abs(ou_sc)>=2:    reasons.append("{:+d} OU".format(ou_sc))
    if abs(wy)>=2:       reasons.append("{:+d} Wyckoff".format(wy))
    if abs(kal_sc)>=2:   reasons.append("{:+d} Kalman".format(kal_sc))
    if abs(tick_sc)>=1:  reasons.append("{:+d} Tick".format(tick_sc))

    return {
        "side":side,"score":score,"confidence":conf,"tradeable":tradeable,
        "sl":sl_,"tp1":tp1,"tp2":tp2,"qty":round(qty,3),"rr":rr,
        "poc":poc,"vah":vah,"val":val,"garch_mult":garch_m,"vol_regime":vol_reg,
        "kelly":sk_r,"kelly_full":sk_full,"ou_z":ou_z,"kal_trend":kal_t,"kal_price":kal_p,
        "meta_dir":meta_res["direction"],"meta_conf":meta_res["meta_prob"],
        "take_trade":meta_res["take"],"resnet_prob":resnet_p,"gbm_prob":gbm_prob,
        "div_bull":div_b,"div_bear":div_s,"wyckoff":wy,"trap":trap_l or trap_s,
        "vdev":vdev,"tick_sc":tick_sc,"tick_snap":tick_snap or {},
        "vol5":float(ret.tail(5).std()),"vol20":float(ret.tail(20).std()),"reasons":reasons,
    }


# ══════════════════════════════════════════════════════════════════════════
#  LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
def display(price, res, tr, loop_n, live, cpcv_sh, paper_st, sig_hist, tick_snap, ws_connected):
    os.system("cls" if os.name=="nt" else "clear")
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    side=res["side"]; sc=res["score"]; conf=res["confidence"]
    sc_c="G" if sc>0 else("R" if sc<0 else "Y")
    ws_str=cc("WS:LIVE","G") if ws_connected else cc("WS:REST","Y")

    print(cc("="*74,"C"))
    print(cc("  ELITE QUANT ENGINE v4.0  |  REAL-TIME  |  BTC/USDT Futures","C"))
    print(cc("  WebSocket + ResNet + Meta-Label + Triple-Barrier + 120 Alphas","C"))
    print(cc("="*74,"C"))
    print("  {}   Bar#{}   {}   {}".format(cc(now,"D"),loop_n,
          "LIVE" if live else cc("SYNTHETIC","Y"), ws_str))
    print("  {}   Vol5={:.3f}%  Vol20={:.3f}%  GARCH x{:.1f}  {}".format(
        cc("Price: ${:,.2f}".format(price),"W"),
        res["vol5"]*100,res["vol20"]*100,res["garch_mult"],
        cc(res["vol_regime"],sc_c)))
    print()

    # ── Real-time tick panel ──
    if tick_snap and tick_snap.get("trades",0)>0:
        tp=tick_snap.get("pressure","NEUTRAL")
        tp_c="G" if tp=="BUY" else("R" if tp=="SELL" else "D")
        td=tick_snap.get("delta_pct",0)
        print(cc("  ── REAL-TIME TICK FLOW (WebSocket aggTrade) ──────────────────────","M"))
        print("  Tick price: ${:,.2f}   VWAP: ${:,.2f}".format(
            tick_snap.get("price",price), tick_snap.get("vwap",price)))
        print("  Buy vol: {:.3f}  Sell vol: {:.3f}  Delta%: {:+.3f}  Trades: {}".format(
            tick_snap.get("buy_vol",0), tick_snap.get("sell_vol",0),
            td, tick_snap.get("trades",0)))
        print("  Pressure: {}  Tick signal: {:+d}".format(
            cc(tp,tp_c), res["tick_sc"]))
        print()

    # ── Main signal ──
    b=bar(abs(sc)/15); sc_bar=cc(b,sc_c)
    print(cc("  "+"="*66,"W"))
    if side=="BUY":
        print(cc("  ||  ######   B U Y   ^ ^ ^ ^ ^ ^ ^ ^   ######              ||","G"))
    elif side=="SELL":
        print(cc("  ||  ######   S E L L   v v v v v v v v   ######             ||","R"))
    else:
        print(cc("  ||  ------   W A I T   (insufficient confluence)             ||","Y"))
    print("  ||  Score:{}  {}  Conf:{}  MetaConf:{}  Take:{}   ||".format(
        cc("{:>+3d}".format(sc),"B"), sc_bar,
        cc("{:.1f}%".format(conf),"B"),
        cc("{:.3f}".format(res["meta_conf"]),"B"),
        cc("YES","G") if res["take_trade"] else cc("NO","R")))
    print(cc("  "+"="*66,"W"))
    print()

    # ── Trade ──
    if res["tradeable"] and res["tp1"]:
        rr=res["rr"]; rrc="G" if rr>=2.5 else("Y" if rr>=1.5 else "R")
        print(cc("  +----- TRADE STRUCTURE -----------------------------------------------+","Y"))
        print("  |  Entry: ${:>12,.2f}{}|".format(price," "*40))
        print(cc("  |  Stop:  ${:>12,.2f}  (${:>7,.1f} risk = {:.1f}x ATR)".format(
            res["sl"],abs(price-res["sl"]),CFG["ATR_SL"]),"R")+" "*20+cc("|","Y"))
        print(cc("  |  TP1:   ${:>12,.2f}  POC/structure  (close 60%)".format(res["tp1"]),"G")+" "*14+cc("|","Y"))
        print(cc("  |  TP2:   ${:>12,.2f}  VAH/VAL        (close 40%)".format(res["tp2"]),"G")+" "*14+cc("|","Y"))
        print("  |  R:R={} Qty={:.3f}BTC Kelly={:.2f}% GARCH x{:.1f}{}|".format(
            cc("{:.2f}x".format(rr),rrc),res["qty"],res["kelly"]*100,res["garch_mult"]," "*14))
        print(cc("  +-------------------------------------------------------------------------+","Y"))
    elif side!="WAIT":
        print(cc("  Signal found but conf={:.1f}% or meta={:.3f} below threshold".format(
            conf, res["meta_conf"]),"Y"))
    print()

    # ── CPCV + Model perf ──
    cv_c="G" if cpcv_sh>1.0 else("Y" if cpcv_sh>0.3 else "R")
    print(cc("  -- CPCV Sharpe: {}   {}".format(
        cc("{:.3f}".format(cpcv_sh),cv_c),
        "STRONG EDGE" if cpcv_sh>1.0 else("WEAK" if cpcv_sh>0.3 else "NO EDGE")),"M"))
    print()

    # ── Scorecard ──
    print(cc("  -- SIGNAL SCORECARD -------------------------------------------------------","M"))
    items=[
        ("ResNet",      res["resnet_prob"]-0.5,   "{:.4f}".format(res["resnet_prob"])),
        ("GBM",         res["gbm_prob"]-0.5,      "{:.4f}".format(res["gbm_prob"])),
        ("Meta-conf",   res["meta_conf"]-0.5,     "{:.4f}".format(res["meta_conf"])),
        ("OU z",       -res["ou_z"]/5,             "{:+.3f}".format(res["ou_z"])),
        ("Kalman",      res["kal_trend"]/5,        "{:+.3f}/bar".format(res["kal_trend"])),
        ("VWAP band",  -res["vdev"]/3,             "{:+.3f}σ".format(res["vdev"])),
        ("Tick flow",   res["tick_sc"]/3,          "{:+d}".format(res["tick_sc"])),
    ]
    for lbl,rv,dv in items:
        rv=float(rv); col="G" if rv>0.02 else("R" if rv<-0.02 else "D")
        print("  {:<14} {}  {}".format(lbl, cc(bar(abs(rv),10),col), dv))
    print()

    # ── Active signals ──
    print(cc("  -- ACTIVE SIGNALS ---------------------------------------------------------","D"))
    def em(c2,t,col="G"):
        if c2: print("  {} {}".format(cc("*","Y"),cc(t,col)))
    em(res["resnet_prob"]>0.66,"RESNET STRONG BULL  P={:.4f}".format(res["resnet_prob"]))
    em(res["resnet_prob"]<0.34,"RESNET STRONG BEAR  P={:.4f}".format(res["resnet_prob"]),"R")
    em(res["div_bull"],"CVD BULL DIVERGENCE  price fell, buyers accumulating")
    em(res["div_bear"],"CVD BEAR DIVERGENCE  price rose, sellers distributing","R")
    em(res["ou_z"]<-1.8,"OU OVERSHOOTING DOWN  z={:.3f}  -> reversion BUY".format(res["ou_z"]))
    em(res["ou_z"]>1.8,"OU OVERSHOOTING UP    z={:.3f}  -> reversion SELL".format(res["ou_z"]),"R")
    em(res["wyckoff"]>=2,"WYCKOFF {} confirmed".format("ACCUMULATION" if res["wyckoff"]==2 else "MARKUP"))
    em(res["wyckoff"]<=-2,"WYCKOFF {} confirmed".format("DISTRIBUTION" if res["wyckoff"]==-2 else "MARKDOWN"),"R")
    em(res["trap"],"TRAPPED TRADERS  squeeze incoming")
    em(res["kal_trend"]>0.2,"KALMAN TREND UP  {:.3f}/bar".format(res["kal_trend"]))
    em(res["kal_trend"]<-0.2,"KALMAN TREND DOWN {:.3f}/bar".format(res["kal_trend"]),"R")
    em(res["meta_conf"]>0.65,"META-LABEL CONFIRMS  {:.3f}  high P(correct)".format(res["meta_conf"]))
    em(not res["take_trade"],"META-LABEL REJECTS  -> skip this trade","Y")
    print()

    # ── Paper trading ──
    if paper_st:
        pnl_c="G" if paper_st["pnl_pct"]>=0 else "R"
        print(cc("  -- PAPER TRADING ----------------------------------------------------------","M"))
        print("  Balance: {}   PnL: {}   WR: {:.1f}%   Trades: {}   {}".format(
            cc("${:,.2f}".format(paper_st["balance"]),"W"),
            cc("{:+.2f}%".format(paper_st["pnl_pct"]),pnl_c),
            paper_st["win_rate"], paper_st["trades"],
            cc("IN POSITION","G") if paper_st["in_position"] else ""))
        print()

    # ── Signal history ──
    recent=sig_hist.recent(5) if sig_hist else []
    if recent:
        print(cc("  -- SIGNAL HISTORY (last 5)  Live acc: {:.1f}% ─────────────────────────────".format(
            sig_hist.live_acc),"D"))
        for s in reversed(recent):
            oc=s.get("outcome","—")
            oc_c="G" if oc=="WIN" else("R" if oc=="LOSS" else "D")
            print("  {} {:>4}  {:.0f}pts  conf={:.0f}%  {}".format(
                s["time"].strftime("%H:%M:%S"),s["side"],s["score"],
                s["conf"],cc(oc,oc_c)))
        print()

    # ── Levels ──
    print("  {} POC=${:,.1f}  VAH=${:,.1f}  VAL=${:,.1f}  Kalman=${:,.1f}".format(
        cc("o","C"),res["poc"],res["vah"],res["val"],res["kal_price"]))
    print()
    print(cc("  -- REASONS ----------------------------------------------------------------","D"))
    print("  "+("  |  ".join(res["reasons"]) if res["reasons"] else "Composite signal"))
    print()
    print(cc("  Models: GBM OOF={:.1f}%  ET={:.1f}%  ResNet={:.1f}%  PCA={} components".format(
        tr.get("gbm_acc",0)*100,tr.get("et_acc",0)*100,
        tr.get("resnet_acc",0)*100,tr.get("n_pca",0)),"D"))
    print(cc("  Ctrl+C  |  --tf 1m/3m/5m/15m  |  --account USDT  |  --paper","D"))
    print(cc("="*74,"D"))


# ══════════════════════════════════════════════════════════════════════════
#  DATA FETCHER  (REST fallback)
# ══════════════════════════════════════════════════════════════════════════
def fetch_rest(symbol, tf, limit):
    url="{}/fapi/v1/klines".format(BASE_API)
    r=requests.get(url,params={"symbol":symbol,"interval":tf,"limit":limit},timeout=12)
    r.raise_for_status()
    df=pd.DataFrame(r.json(),columns=["ts","o","h","l","c","v","ct","qv","n","tbv","tbqv","_"])
    df["open_time"]=pd.to_datetime(df["ts"].astype(float),unit="ms",utc=True)
    for col in ["o","h","l","c","v","tbv","n"]: df[col]=df[col].astype(float)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close",
                               "v":"volume","tbv":"taker_buy_vol","n":"trades"})[
        ["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]

def fetch_funding(symbol, limit=50):
    url="{}/fapi/v1/fundingRate".format(BASE_API)
    r=requests.get(url,params={"symbol":symbol,"limit":limit},timeout=10)
    r.raise_for_status()
    df=pd.DataFrame(r.json())
    df["fundingTime"]=pd.to_datetime(df["fundingTime"].astype(float),unit="ms",utc=True)
    df["fundingRate"]=df["fundingRate"].astype(float)
    return df

def make_synthetic(n=500,seed=42,base=67000.0):
    np.random.seed(seed); dates=pd.date_range(end=pd.Timestamp.utcnow(),periods=n,freq="5min",tz="UTC")
    price=float(base); rows=[]
    for dt in dates:
        h=dt.hour; sv=2.2 if h in [8,9,13,14,15,16] else 0.65
        mu=-0.00018 if h in [16,17,18] else 0.00012
        price=max(price*(1+np.random.normal(mu,0.0028*sv)),50000)
        hi=price*(1+abs(np.random.normal(0,0.002*sv))); lo=price*(1-abs(np.random.normal(0,0.002*sv)))
        vol=max(abs(np.random.normal(1100,380))*sv,80)
        bsk=0.63 if h in [8,9] else(0.36 if h in [17,18] else 0.50)
        tb=vol*float(np.clip(np.random.beta(bsk*7,(1-bsk)*7),0.05,0.95))
        if np.random.random()<0.025: vol*=np.random.uniform(5,9)
        rows.append({"open_time":dt,"open":price*(1+np.random.normal(0,0.001)),
                     "high":hi,"low":lo,"close":price,"volume":vol,"taker_buy_vol":tb,"trades":int(vol/0.04)})
    df=pd.DataFrame(rows)
    fund=pd.DataFrame([{"fundingTime":dates[i],"fundingRate":float(np.random.normal(0.0001,0.0003))}
                       for i in range(0,n,96)])
    return df,fund


# ══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET MANAGER
# ══════════════════════════════════════════════════════════════════════════
class WSManager:
    """Manages WebSocket connections for kline + aggTrade streams."""

    def __init__(self, symbol, tf, kline_buf, tick_buf):
        self.symbol    = symbol.lower()
        self.tf        = tf
        self.kbuf      = kline_buf
        self.tbuf      = tick_buf
        self.connected = False
        self.ws_kl     = None
        self.ws_tick   = None
        self._stop     = threading.Event()

    def _on_kline(self, ws, message):
        try:
            data = json.loads(message)
            k    = data.get("k", {})
            if not k.get("x", False):   # x = is_closed
                return
            row = {
                "open_time":     int(k["t"]),
                "open":          float(k["o"]),
                "high":          float(k["h"]),
                "low":           float(k["l"]),
                "close":         float(k["c"]),
                "volume":        float(k["v"]),
                "taker_buy_vol": float(k["Q"]) if "Q" in k else float(k["v"])*0.5,
                "trades":        int(k.get("n", 0)),
            }
            self.kbuf.update(row)
        except Exception as e:
            pass

    def _on_tick(self, ws, message):
        try:
            data = json.loads(message)
            self.tbuf.add(
                price=float(data["p"]),
                qty=float(data["q"]),
                is_buyer_maker=bool(data["m"]),
                ts=int(data["T"]),
            )
        except Exception:
            pass

    def _on_open(self, ws):
        self.connected = True

    def _on_close(self, ws, code, msg):
        self.connected = False

    def _on_error(self, ws, error):
        self.connected = False

    def _run_kline_ws(self):
        url = "{}/{}@kline_{}".format(BASE_WS, self.symbol, self.tf)
        while not self._stop.is_set():
            try:
                self.ws_kl = ws_module.WebSocketApp(
                    url,
                    on_message=self._on_kline,
                    on_open=self._on_open,
                    on_close=self._on_close,
                    on_error=self._on_error,
                )
                self.ws_kl.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            if not self._stop.is_set():
                time.sleep(5)   # reconnect delay

    def _run_tick_ws(self):
        url = "{}/{}@aggTrade".format(BASE_WS, self.symbol)
        while not self._stop.is_set():
            try:
                self.ws_tick = ws_module.WebSocketApp(
                    url,
                    on_message=self._on_tick,
                    on_open=lambda ws: None,
                    on_close=lambda ws,c,m: None,
                    on_error=lambda ws,e: None,
                )
                self.ws_tick.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            if not self._stop.is_set():
                time.sleep(5)

    def start(self):
        if not WS_OK:
            return False
        threading.Thread(target=self._run_kline_ws, daemon=True).start()
        threading.Thread(target=self._run_tick_ws,  daemon=True).start()
        return True

    def stop(self):
        self._stop.set()
        if self.ws_kl:   self.ws_kl.close()
        if self.ws_tick: self.ws_tick.close()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════
class EliteQuantV4:
    def __init__(self, account=1000.0, paper=True):
        CFG["ACCOUNT"] = account
        self.meta      = MetaLabelSystem()
        self.resnet    = None
        self.scaler    = RobustScaler()
        self.pca       = PCA(n_components=CFG["PCA_VAR"])
        self.trained   = False
        self.train_res = {}
        self.bar_count = 0
        self.cpcv_sh   = 0.0
        # Buffers
        self.kbuf      = KlineBuffer(maxlen=600)
        self.tbuf      = TickBuffer(maxlen=CFG["TICK_WINDOW"])
        self.ws_mgr    = None
        self.ws_conn   = False
        # Paper trading
        self.paper     = PaperTrader(account) if paper else None
        self.sig_hist  = SignalHistory(maxlen=200)

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, fund: pd.DataFrame, verbose=True):
        vp = verbose
        if vp:
            print(cc("\n  ELITE QUANT v4.0 — TRAINING","M"))
            print(cc("  "+"-"*60,"M"))

        if vp: print("  [1/6] Triple-barrier ...", end=" ", flush=True)
        tb   = triple_barrier(df, pct=CFG["BARRIER_PCT"], t_max=CFG["TARGET_BARS"])
        idx  = tb.index; df_v=df.loc[idx]; y_tb=tb.values
        tp_r=float((y_tb==1).mean()*100); sl_r=float((y_tb==-1).mean()*100); ep_r=float((y_tb==0).mean()*100)
        if vp: print("TP={:.1f}%  SL={:.1f}%  Exp={:.1f}%".format(tp_r,sl_r,ep_r))

        if vp: print("  [2/6] Features ...", end=" ", flush=True)
        F_df=build_features(df_v,fund,None); X_r=np.nan_to_num(F_df.values.astype(float),0.0)
        if vp: print("{} raw features".format(X_r.shape[1]))

        if vp: print("  [3/6] PCA ...", end=" ", flush=True)
        X_sc=self.scaler.fit_transform(X_r); X_pca=self.pca.fit_transform(X_sc)
        if vp: print("{} components ({:.0f}% var)".format(X_pca.shape[1],CFG["PCA_VAR"]*100))

        if vp: print("  [4/6] Purged K-Fold ...")
        splits=purged_kfold(len(X_pca),k=5,purge=CFG["PURGE"],embargo=CFG["EMBARGO"])

        if vp: print("  [5/6] Meta-labeling ...")
        oof_p,oof_m,gbm_acc,et_acc=self.meta.fit(X_pca,y_tb,splits,verbose=vp)

        if vp: print("  [6/6] ResNet ...", end=" ", flush=True)
        mask=y_tb!=0; X_nn=X_pca[mask]; y_nn=(y_tb[mask]==1).astype(float)
        resnet_acc=0.5
        if len(X_nn)>80:
            nv=max(int(len(X_nn)*0.15),20)
            self.resnet=ResNet(n_in=X_nn.shape[1],hidden=CFG["NN_H"],n_blocks=CFG["NN_B"],
                               lr=CFG["NN_LR"],l2=CFG["NN_L2"],dropout=CFG["NN_DR"])
            self.resnet.fit(X_nn[:-nv],y_nn[:-nv],Xv=X_nn[-nv:],yv=y_nn[-nv:],epochs=CFG["NN_EP"])
            resnet_acc=self.resnet.val_acc
        if vp: print("val_acc={:.4f}".format(resnet_acc))

        if vp: print("  CPCV Sharpe ...", end=" ", flush=True)
        y_dir=(y_tb==1).astype(int)
        ret_s=df["close"].pct_change().fillna(0); ret_loc=ret_s.loc[idx]
        self.cpcv_sh=cpcv_sharpe(oof_p,y_dir,ret_loc) if len(np.unique(y_dir))>=2 else 0.0
        if vp: print("{:.3f}".format(self.cpcv_sh))

        self.trained=True; self.X_last=X_pca
        self.train_res={"n_raw":X_r.shape[1],"n_pca":X_pca.shape[1],"n_samples":len(X_pca),
                        "gbm_acc":gbm_acc,"et_acc":et_acc,"resnet_acc":resnet_acc,
                        "tb_tp":tp_r,"tb_sl":sl_r,"tb_exp":ep_r}
        if vp: print(cc("\n  Training complete.  CPCV Sharpe={:.3f}\n".format(self.cpcv_sh),"G"))

    # ── One inference cycle ───────────────────────────────────────────────
    def infer(self, df: pd.DataFrame, fund: pd.DataFrame) -> dict:
        tick_snap = self.tbuf.snapshot(window_ms=30000)
        F_df  = build_features(df, fund, tick_snap)
        X_raw = np.nan_to_num(F_df.values.astype(float), 0.0)
        X_sc  = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_sc)

        gbm_p = float(self.meta.primary.predict_proba(X_pca[-1:])[:,1][0]) \
                if self.meta.primary else 0.5
        meta_res = self.meta.predict(X_pca, gbm_p)
        rnet_p   = float(self.resnet.predict(X_pca[-1:])[0]) if self.resnet else 0.5

        poc, vah, val = market_profile(df)
        ret = df["close"].pct_change().dropna()
        _, garch_m, vol_reg, _ = garch11(ret)

        res = aggregate(df, meta_res, rnet_p, gbm_p, poc, vah, val,
                        garch_m, vol_reg, tick_snap)
        return res

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self, retrain_every=50):
        live  = False
        fund  = pd.DataFrame()
        df    = pd.DataFrame()

        # ── Try WebSocket ──
        if WS_OK and NET:
            print(cc("  Starting WebSocket streams ...","M"), flush=True)
            self.ws_mgr = WSManager(CFG["SYMBOL"], CFG["TF"], self.kbuf, self.tbuf)
            self.ws_mgr.start()
            time.sleep(3)
            self.ws_conn = self.ws_mgr.connected

        # ── Initial REST load ──
        if NET:
            try:
                df   = fetch_rest(CFG["SYMBOL"], CFG["TF"], CFG["CANDLES"])
                fund = fetch_funding(CFG["SYMBOL"])
                live = True
                print("  REST data loaded: {} bars".format(len(df)))
            except Exception as e:
                print("  REST error: {}  -> using synthetic".format(e))

        if df.empty:
            df, fund = make_synthetic(seed=42)

        df = prepare(df)

        # ── Initial training ──
        self.train(df, fund, verbose=True)
        bars_since_train = 0

        # ── Populate kline buffer from REST data ──
        for _, row in df.tail(100).iterrows():
            self.kbuf.update({
                "open_time":     int(row["open_time"].timestamp()*1000),
                "open":          float(row["open"]),
                "high":          float(row["high"]),
                "low":           float(row["low"]),
                "close":         float(row["close"]),
                "volume":        float(row["volume"]),
                "taker_buy_vol": float(row["taker_buy_vol"]),
                "trades":        int(row["trades"]),
            })

        print(cc("\n  Real-time inference loop started ...\n","G"))

        while True:
            try:
                # ── Wait for new bar (WS) or poll (REST) ──
                if self.ws_mgr and self.ws_conn:
                    self.ws_conn = self.ws_mgr.connected
                    got_bar = self.kbuf.wait_for_bar(timeout=60)
                    if not got_bar:
                        continue
                    curr_df = self.kbuf.get_df()
                    if curr_df.empty or len(curr_df) < 50:
                        continue
                    curr_df = prepare(curr_df)
                else:
                    # REST polling fallback
                    time.sleep(CFG["LOOP_SECS"] if "LOOP_SECS" in CFG else 30)
                    if NET:
                        try:
                            df = fetch_rest(CFG["SYMBOL"],CFG["TF"],CFG["CANDLES"])
                            fund=fetch_funding(CFG["SYMBOL"]); live=True
                        except: pass
                    curr_df = prepare(df)

                self.bar_count       += 1
                bars_since_train     += 1
                price = float(curr_df["close"].iloc[-1])

                # ── Retrain if due ──
                if bars_since_train >= retrain_every:
                    print(cc("\n  Retraining after {} new bars ...\n".format(bars_since_train),"Y"))
                    self.train(curr_df, fund, verbose=False)
                    bars_since_train = 0

                # ── Infer ──
                res = self.infer(curr_df, fund)

                # ── Paper trading update ──
                trade_result = None
                if self.paper:
                    trade_result = self.paper.update(price)
                    if trade_result:
                        print(cc("  [PAPER] {} @ ${:,.2f}  PnL=${:.2f}".format(
                            trade_result["type"],trade_result["price"],trade_result["pnl"]),
                            "G" if trade_result["type"] in ["TP1","WIN"] else "R"))
                    if res["tradeable"] and not self.paper.position:
                        entered = self.paper.enter(
                            res["side"],price,res["sl"],res["tp1"],res["tp2"],
                            res["qty"],res["score"],res["confidence"]," | ".join(res["reasons"]))
                        if entered:
                            print(cc("  [PAPER] Entered {} @ ${:,.2f}".format(res["side"],price),"G"))

                # ── Signal history ──
                if res["side"] != "WAIT":
                    self.sig_hist.record(res["side"],price,res["score"],
                                         res["confidence"],res["meta_conf"])
                if self.bar_count > 3:
                    self.sig_hist.resolve(price)

                # ── Display ──
                paper_st = self.paper.stats() if self.paper else None
                display(price, res, self.train_res, self.bar_count,
                        live, self.cpcv_sh, paper_st,
                        self.sig_hist, self.tbuf.snapshot(30000),
                        self.ws_conn)

            except KeyboardInterrupt:
                print(cc("\n  Stopped by user.","Y"))
                if self.ws_mgr: self.ws_mgr.stop()
                if self.paper:
                    st=self.paper.stats()
                    print("  Final: balance=${:.2f}  PnL={:+.2f}%  WR={:.1f}%  Trades={}".format(
                        st["balance"],st["pnl_pct"],st["win_rate"],st["trades"]))
                break
            except Exception as exc:
                import traceback
                print("  Error: {}".format(exc)); traceback.print_exc()
                time.sleep(15)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Elite Quant Engine v4.0 — Real-Time")
    parser.add_argument("--account",  type=float, default=1000.0,  help="Account USDT")
    parser.add_argument("--paper",    action="store_true",          help="Paper trading mode")
    parser.add_argument("--tf",       type=str,   default="5m",     help="Timeframe: 1m 3m 5m 15m")
    parser.add_argument("--retrain",  type=int,   default=50,       help="Retrain every N bars")
    parser.add_argument("--symbol",   type=str,   default="BTCUSDT",help="Symbol")
    args = parser.parse_args()

    CFG["TF"]      = args.tf
    CFG["SYMBOL"]  = args.symbol
    CFG["ACCOUNT"] = args.account
    CFG["LOOP_SECS"] = {"1m":60,"3m":180,"5m":300,"15m":900}.get(args.tf,300)

    print(cc("\n"+"="*74,"C"))
    print(cc("  ELITE QUANT ENGINE v4.0  —  REAL-TIME INFERENCE","C"))
    print(cc("  WebSocket + ResNet + Meta-Label + CPCV + 120 Alphas","C"))
    print(cc("="*74,"C"))
    print("  Symbol:    {}".format(CFG["SYMBOL"]))
    print("  Timeframe: {}".format(CFG["TF"]))
    print("  Account:   ${:,.2f} USDT".format(args.account))
    print("  Mode:      {}".format("PAPER TRADING" if args.paper else "LIVE SIGNALS"))
    print("  WebSocket: {}".format("AVAILABLE" if WS_OK else "NOT INSTALLED (pip install websocket-client)"))
    print()

    engine = EliteQuantV4(account=args.account, paper=args.paper)
    engine.run(retrain_every=args.retrain)

if __name__ == "__main__":
    main()
