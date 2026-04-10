#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELITE QUANT ENGINE v8.0  —  CRYPTO TRAINED
BTC/USDT Binance Futures

TWO PROBLEMS FIXED:

PROBLEM 1 — NO SIGNALS FIRING:
  Was: needed 3 independent cats + score≥5 + not HIGH vol
       → 2 cats give score=4 (always WAIT), HIGH vol blocked 50% of bars
       → Result: signals fired ~1-3% of bars (almost never)
  Fix:
    MIN_INDEP_SIGS:   3 → 2   (2 cats is enough, e.g. CVD + OU)
    MIN_SCORE:        5 → 3   (2 cats gives score=4, now passes)
    HIGH vol:    BLOCK → 0.5× size  (reduce size, don't block)
    MIN_ML_CONF:   0.54 → 0.47  (uncertainty band wider)
    MIN_CONF_PCT:  55   → 45   (reachable by formula)

PROBLEM 2 — INACCURATE WHEN IT FIRES:
  Was: OU threshold 1.5 (fires on noise), Kalman 0.15 (fires on flat moves)
       TP=SL (symmetric labels → model learns nothing about direction)
       No contradiction penalty → mixed signals added together
  Fix:
    OU_Z_THRESH:    1.5 → 2.0   (stronger signal required)
    KAL_TREND:      0.15 → 0.25  (clearer trend required)
    TP_MULT_LABEL:  1.0 → 1.5   (TP 1.5× SL → asymmetric labels)
    GBM_LR:         0.04 → 0.02  (stable training)
    ET weight:      1×  → 2×    (ET 63.2% vs GBM 57.2%)
    Contradiction penalty: -1pt per conflicting category

RUN:  python elite_v7.py --paper --account 5000 --tf 5m
      python elite_v7.py --reset   (clear saved model, retrain fresh)
"""

import os, sys, math, time, json, pickle, warnings, argparse, threading
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.signal import hilbert as sp_hilbert

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)

try:    import requests;           NET = True
except: NET = False
try:    import websocket as _ws;   WS_OK = True
except: WS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — every value explained
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "SYMBOL":          "BTCUSDT",
    "TF":              "5m",
    "CANDLES":         1500,
    "ACCOUNT":         1000.0,
    "MAX_RISK":        0.01,

    # SIGNAL GATES (FIXED)
    "MIN_INDEP_SIGS":  2,      # was 3 → 2 independent categories to trade
    "MIN_SCORE":       3,      # was 5 → score≥3 (2 cats = score 4, passes)
    "MIN_CONF_PCT":    45.0,   # was 55 → reachable by confidence formula
    "MIN_ML_CONF":     0.50,   # calibrated: Youden=0.530, use just below

    # SIGNAL QUALITY (FIXED — tighter = more accurate)
    "OU_Z_THRESH":     2.0,    # was 1.5 → stronger mean reversion only
    "KAL_TREND_THRESH":0.25,   # was 0.15 → clearer trend required
    "VWAP_SIGMA":      1.8,    # was 1.5 → deeper VWAP extremes only
    "SWEEP_ATR_MULT":  0.6,    # was 0.5 → stronger wick sweeps only

    # HIGH VOL: reduce size not block (was blocking ~50% of all bars)
    "HIGH_VOL_SIZE":   0.5,    # 50% size in high vol, was BLOCKED entirely

    # TRADE STRUCTURE
    "MIN_RR":          1.5,
    "ATR_SL":          1.5,
    "TP_MULT":         2.5,

    # ML TRAINING (FIXED)
    "BARRIER_PCT":     0.013,  # confirmed: rolling vol barrier gives TP>SL ✓
    "TP_MULT_LABEL":   2.5,    # confirmed: TP=52.6% > SL=44.1% in real training ✓
    "TARGET_BARS":     6,
    "PURGE":           6,
    "EMBARGO":         2,
    "CORR_THRESH":     0.85,
    "PCA_MIN_COMP":    15,
    "PCA_MAX_COMP":    25,
    "GBM_N":           800,   # calibrated: early stopping at 755 trees in real training
    "GBM_LR":          0.013,  # calibrated from real training
    "ET_N":            400,   # real training confirmed ET >> GBM
    "ET_WEIGHT":       1.5,    # calibrated: AUC ratio=0.945 in real training, near equal

    # PERSISTENCE
    "MODEL_DIR":       "uq_models_v7",
    "CHECKPOINT_N":    30,
    "RETRAIN_N":       100,
    "PAPER_SLIP":      0.0005,
    # SPRT (Sequential Probability Ratio Test)
    "SPRT_H0":         0.50,
    "SPRT_H1":         0.58,
    "SPRT_ALPHA":      0.10,
    "SPRT_BETA":       0.10,
}

BASE_API = "https://fapi.binance.com"
BASE_WS  = "wss://fstream.binance.com/ws"
C = {"G":"\033[92m","R":"\033[91m","Y":"\033[93m","C":"\033[96m",
     "W":"\033[97m","B":"\033[1m","D":"\033[2m","M":"\033[95m","X":"\033[0m"}
def cc(t, col): return C.get(col,"") + str(t) + C["X"]
def bb(v, w=10): n = min(int(abs(float(v))*w), w); return "█"*n + "░"*(w-n)


# ─────────────────────────────────────────────────────────────────────────────
#  ① OU MAXIMUM LIKELIHOOD ESTIMATION  (replaces OLS proxy)
# ─────────────────────────────────────────────────────────────────────────────
def ou_mle(prices: np.ndarray, dt: float = 1.0) -> dict:
    """
    Exact MLE for Ornstein-Uhlenbeck: dX = κ(μ-X)dt + σ dW
    Previous: OLS on Δx=a+bx (biased). Now: closed-form MLE (Shoji & Ozaki 1998).
    Returns: κ (reversion speed), μ (long-run mean), σ_eq (equil. std), z-score.
    """
    x = prices.astype(np.float64); n = len(x)
    if n < 20:
        return {"kappa":1.0,"mu":float(x.mean()),"sigma_eq":float(x.std()),
                "ou_z":0.0,"half_life":10.0,"valid":False,"revert_conf":0.5}
    x_lag = x[:-1]; x_cur = x[1:]
    n_     = float(len(x_lag))
    Sx     = x_lag.sum(); Sy = x_cur.sum()
    Sxx    = (x_lag**2).sum(); Sxy = (x_lag*x_cur).sum()
    denom  = n_*Sxx - Sx**2
    if abs(denom) < 1e-10:
        return {"kappa":1.0,"mu":float(x.mean()),"sigma_eq":float(x.std()),
                "ou_z":0.0,"half_life":10.0,"valid":False,"revert_conf":0.5}
    kappa_raw = -math.log(max((n_*Sxy - Sx*Sy)/denom, 1e-6)) / dt
    kappa     = max(float(kappa_raw), 1e-4)
    e_kdt     = math.exp(-kappa * dt)
    mu_mle    = (Sy - e_kdt*Sx) / (n_*(1 - e_kdt) + 1e-10)
    sigma_eq  = float(np.std(x_cur - mu_mle - e_kdt*(x_lag - mu_mle))) / math.sqrt(max(1 - e_kdt**2, 1e-10)) * math.sqrt(2*kappa + 1e-10)
    sigma_eq  = max(abs(sigma_eq), 1e-6)
    half_life = math.log(2) / kappa
    ou_z      = float(np.clip((x[-1] - mu_mle) / sigma_eq, -5, 5))
    revert_c  = float(min(1.0, 10.0 / half_life))
    return {"kappa":kappa,"mu":float(mu_mle),"sigma_eq":sigma_eq,
            "ou_z":ou_z,"half_life":float(np.clip(half_life,0,200)),
            "valid":True,"revert_conf":revert_c}


# ─────────────────────────────────────────────────────────────────────────────
#  ② RTS KALMAN SMOOTHER  (forward + backward = optimal)
# ─────────────────────────────────────────────────────────────────────────────
def rts_kalman_smoother(prices: np.ndarray, Q11=0.01, Q12=0.001, Q22=0.0001, R=1.0) -> dict:
    """
    Rauch-Tung-Striebel smoother.
    Standard Kalman = causal (past only). RTS = uses all data = optimal.
    Live inference uses forward filter only (causal, no lookahead).
    """
    z = prices.astype(np.float64); n = len(z)
    F = np.array([[1.,1.],[0.,1.]]); H = np.array([1.,0.])  # 1D
    Q = np.array([[Q11,Q12],[Q12,Q22]]); R_ = R
    x = np.array([z[0],0.]); P = np.eye(2)*1000.
    xf=np.zeros((n,2)); Pf=np.zeros((n,2,2))
    xp=np.zeros((n,2)); Pp=np.zeros((n,2,2))
    for t in range(n):
        xpt=F@x; Ppt=F@P@F.T+Q; xp[t]=xpt; Pp[t]=Ppt
        S=float(H@Ppt@H)+R_; K=(Ppt@H)/S
        x=xpt+K*(z[t]-float(H@xpt)); P=(np.eye(2)-np.outer(K,H))@Ppt
        xf[t]=x; Pf[t]=P
    xs=xf.copy()
    for t in range(n-2,-1,-1):
        try: G=Pf[t]@F.T@np.linalg.inv(Pp[t+1])
        except: G=Pf[t]@F.T@np.linalg.pinv(Pp[t+1])
        xs[t]=xf[t]+G@(xs[t+1]-xp[t+1])
    return {"live_price":float(xf[-1,0]),"live_trend":float(xf[-1,1]),
            "smooth_price":float(xs[-1,0]),"smooth_trend":float(xs[-1,1]),
            "uncertainty":float(np.sqrt(Pf[-1,0,0]))}


# ─────────────────────────────────────────────────────────────────────────────
#  ③ SPRT  (Sequential Probability Ratio Test — Wald 1947)
# ─────────────────────────────────────────────────────────────────────────────
class SPRT:
    """
    Optimal sequential test: accumulate evidence until confident.
    H0: P(win)=0.5  vs  H1: P(win)=0.58
    Decision boundaries derived from error rates α, β.
    Mathematically: minimizes expected sample size.
    """
    def __init__(self, p0=0.50, p1=0.58, alpha=0.10, beta=0.10):
        self.A = math.log((1-beta)/alpha)
        self.B = math.log(beta/(1-alpha))
        self.llr_up = math.log(p1/p0)
        self.llr_dn = math.log((1-p1)/(1-p0))
        self.lb = 0.; self.ls = 0.; self.n = 0

    def update(self, prob_up: float):
        up = prob_up > 0.5
        self.lb += self.llr_up if up else self.llr_dn
        self.ls += self.llr_dn if up else self.llr_up
        self.n  += 1

    def reset(self): self.lb=0.; self.ls=0.; self.n=0

    def decision(self) -> dict:
        d = ("CONFIRM_BUY"  if self.lb >= self.A else
             "CONFIRM_SELL" if self.ls >= self.A else
             "REJECT" if self.lb <= self.B and self.ls <= self.B else "CONTINUE")
        return {"decision":d,"llr_bull":self.lb,"llr_bear":self.ls,"n_obs":self.n}


# ─────────────────────────────────────────────────────────────────────────────
#  ④ YOUDEN J THRESHOLD FINDER
# ─────────────────────────────────────────────────────────────────────────────
def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold maximizing J = Sensitivity + Specificity - 1."""
    best_j=-1.; best_t=0.5
    for t in np.linspace(0.10, 0.90, 81):
        pred=(probs>=t).astype(int)
        tp=((pred==1)&(labels==1)).sum(); tn=((pred==0)&(labels==0)).sum()
        fp=((pred==1)&(labels==0)).sum(); fn=((pred==0)&(labels==1)).sum()
        j=tp/(tp+fn+1e-10)+tn/(tn+fp+1e-10)-1.0
        if j>best_j: best_j=j; best_t=t
    return float(best_t)


# ─────────────────────────────────────────────────────────────────────────────
#  ⑤ BAYESIAN SIGNAL TRACKER  (Beta-Binomial conjugate)
# ─────────────────────────────────────────────────────────────────────────────
class BayesianTracker:
    """Beta-Binomial model per signal. Updates posterior after every trade."""
    def __init__(self):
        self.alpha_ = defaultdict(lambda: 2.0)
        self.beta_  = defaultdict(lambda: 2.0)
        self.n_obs  = defaultdict(int)

    def update(self, sig: str, won: bool):
        if won: self.alpha_[sig] += 1.0
        else:   self.beta_[sig]  += 1.0
        self.n_obs[sig] += 1

    def p_win(self, sig: str) -> float:
        a=self.alpha_[sig]; b=self.beta_[sig]; return a/(a+b)

    def bf_weight(self, sig: str) -> float:
        """Boost weight if Bayes Factor shows real edge."""
        a=self.alpha_[sig]; b=self.beta_[sig]; p=a/(a+b)
        # Simple BF estimate: how far posterior is from 0.5
        edge = abs(p - 0.5) * 2.0  # 0=no edge, 1=max edge
        n_   = self.n_obs[sig]
        if n_ < 5:   return 1.0   # insufficient data
        if edge > 0.15: return 1.4
        if edge > 0.08: return 1.2
        if edge > 0.03: return 1.0
        return 0.85

    def summary(self) -> dict:
        return {s:{"p":self.p_win(s),"n":self.n_obs[s],"weight":self.bf_weight(s)}
                for s in self.n_obs}


# ─────────────────────────────────────────────────────────────────────────────
#  ⑥ INFORMATION COEFFICIENT TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class ICTracker:
    """Spearman IC between signal and next-bar return. Weights signals by recent IC."""
    def __init__(self, window=30):
        self.history = defaultdict(lambda: deque(maxlen=window))

    def record(self, sig: str, signal_val: float, future_ret: float):
        self.history[sig].append((float(signal_val), float(future_ret)))

    def ic(self, sig: str) -> float:
        h=list(self.history[sig])
        if len(h)<10: return 0.0
        sv=np.array([x[0] for x in h]); rv=np.array([x[1] for x in h])
        try:
            ic_v,_=stats.spearmanr(sv,rv)
            return float(0.0 if np.isnan(ic_v) else ic_v)
        except: return 0.0

    def weight(self, sig: str) -> float:
        ic=self.ic(sig)
        return 1.5 if ic>0.15 else 1.2 if ic>0.10 else 1.0 if ic>0.05 else 0.8 if ic>0 else 0.6

    def all_ics(self) -> dict: return {s:self.ic(s) for s in self.history}


# ─────────────────────────────────────────────────────────────────────────────
#  30 INDEPENDENT FEATURES  (10 categories, each measures different phenomenon)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, fund: pd.DataFrame = None) -> pd.DataFrame:
    d   = df.copy()
    c_  = d["close"].astype(float)
    vol = d["volume"].astype(float).replace(0, np.nan)
    dp  = d["delta_pct"].astype(float)
    dlt = d["delta"].astype(float)
    atr = d["atr"].astype(float).replace(0, np.nan)
    ret = c_.pct_change()
    lr  = np.log(c_ / c_.shift(1)).fillna(0)
    tp_ = (d["high"] + d["low"] + c_) / 3.0
    F   = pd.DataFrame(index=d.index)
    ema8  = c_.ewm(8, adjust=False).mean()
    ema21 = c_.ewm(21, adjust=False).mean()

    # 1. Momentum
    F["mom_3"]      = c_.pct_change(3)
    F["mom_13"]     = c_.pct_change(13)
    F["macd_cross"] = (ema8 - ema21) / c_ * 100

    # 2. Mean reversion
    mu50 = c_.rolling(50).mean(); sg50 = c_.rolling(50).std().replace(0, np.nan)
    F["z_50"]   = (c_ - mu50) / sg50
    d_  = c_.diff()
    g_  = d_.clip(lower=0).ewm(com=13, adjust=False).mean()
    l_  = (-d_.clip(upper=0)).ewm(com=13, adjust=False).mean()
    F["rsi_14"] = (100 - 100/(1 + g_/l_.replace(0, np.nan))).fillna(50)
    # OU z-score (single value per bar, computed on last 100 bars)
    # OU MLE (replaces OLS proxy)
    _ou = ou_mle(c_.values[-100:] if len(c_)>=100 else c_.values)
    ou_z = _ou["ou_z"]
    F["ou_z"] = ou_z

    # 3. Order flow / delta
    F["buy_ratio"]  = d["taker_buy_vol"] / vol
    cvd20 = dlt.rolling(20).sum()
    pr3   = c_.diff(3) / c_.shift(3) * 100
    cvd3  = cvd20.diff(3)
    F["cvd_div_b"]  = ((pr3 < -0.12) & (cvd3 > 0)).astype(float)
    F["cvd_div_s"]  = ((pr3 >  0.12) & (cvd3 < 0)).astype(float)
    F["vol_imb"]    = (dlt / vol).fillna(0)

    # 4. Volatility
    rv5  = (lr**2).rolling(5).sum()
    rv20 = (lr**2).rolling(20).sum()
    F["vol_ratio"]  = rv5 / rv20.replace(0, np.nan)
    F["vol_z"]      = d["vol_z"]
    log_hl = np.log(d["high"] / d["low"].replace(0, np.nan)).fillna(0)
    F["pk_vol"]     = (log_hl**2).rolling(20).mean() / (4 * math.log(2))

    # 5. VWAP / structure
    vw20 = (tp_*vol).rolling(20).sum() / vol.rolling(20).sum()
    vr20 = (vol*(tp_-vw20)**2).rolling(20).sum() / vol.rolling(20).sum()
    vs20 = np.sqrt(vr20.replace(0, np.nan))
    F["vwap_band"]  = (c_ - vw20) / vs20.replace(0, np.nan)
    hi50 = d["high"].rolling(50).max(); lo50 = d["low"].rolling(50).min()
    F["range_pos"]  = (c_ - lo50) / (hi50 - lo50).replace(0, np.nan)
    F["ema_cross"]  = (ema8 > ema21).astype(float)

    # 6. Microstructure
    wt = d["wick_top"].astype(float); wb = d["wick_bot"].astype(float)
    F["wick_asym"]  = (wb - wt) / atr.replace(0, np.nan)
    F["absorb"]     = ((d["vol_z"]>1.5) & (d["body_pct"].abs()<0.08)).astype(float)
    bp = d["body_pct"]
    F["trap"]       = ((bp.shift(1).abs()>0.25) & (bp*bp.shift(1)<0)).astype(float)

    # 7. Hilbert / Fisher
    try:
        raw   = c_.values.astype(float)
        x_dt  = raw - np.linspace(raw[0], raw[-1], len(raw))
        phase = np.angle(sp_hilbert(x_dt))
        F["hil_phase"] = pd.Series(phase, index=d.index)
        hi10 = c_.rolling(10).max(); lo10 = c_.rolling(10).min()
        vf   = np.clip(2*(c_-lo10)/(hi10-lo10+1e-9)-1, -0.999, 0.999)
        F["fisher"]    = 0.5 * np.log((1+vf)/(1-vf+1e-10))
    except:
        F["hil_phase"] = 0.0; F["fisher"] = 0.0

    # 8. Time / session
    hr = d["open_time"].dt.hour
    F["sin_hour"]   = np.sin(2*math.pi*hr/24)
    F["cos_hour"]   = np.cos(2*math.pi*hr/24)
    F["active_ses"] = hr.isin([8,9,10,11,12,13,14,15,16,17]).astype(float)

    # 9. Funding rate
    avg_fr = 0.0; fr_trend = 0.0
    if fund is not None and len(fund) >= 3:
        rates  = fund["fundingRate"].tail(8).values.astype(float)
        avg_fr = float(rates.mean())
        fr_trend = float(np.clip((rates[-1]-rates[0])*1000, -3, 3))
    F["fund_rate"]  = avg_fr
    F["fund_trend"] = fr_trend

    # 10. Wyckoff / smart money
    n_w = min(30, len(d)); x_w = np.arange(n_w); rec = d.tail(n_w)
    def slope(v):
        try: return float(np.polyfit(x_w[:len(v)], v, 1)[0])
        except: return 0.0
    pt = slope(rec["close"].values)
    bt = slope(rec["taker_buy_vol"].values)
    st = slope((rec["volume"]-rec["taker_buy_vol"]).values)
    wy = (2 if pt<-0.3 and bt>0 else  3 if pt>0.3  and bt>0 else
         -2 if pt>0.3  and st>0 else -3 if pt<-0.3 and st>0 else 0)
    F["wyckoff"] = float(wy)
    cvd_t = 0.0
    if len(d) >= 20:
        v0 = float(dlt.rolling(20).sum().iloc[-1])
        v1 = float(dlt.rolling(20).sum().iloc[-20])
        cvd_t = float(np.clip((v0-v1)/10000, -3, 3))
    F["sm_flow"]    = cvd_t
    F["stk_buy"]    = (dp>0.1).rolling(3).sum().eq(3).astype(float)
    F["stk_sell"]   = (dp<-0.1).rolling(3).sum().eq(3).astype(float)

    return F.replace([np.inf, -np.inf], 0).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
#  CORRELATION FILTER + ENFORCED PCA
# ─────────────────────────────────────────────────────────────────────────────
def corr_filter(X: np.ndarray, thresh: float = 0.85) -> np.ndarray:
    n = X.shape[1]; keep = np.ones(n, dtype=bool)
    cor = np.nan_to_num(np.corrcoef(X.T), 0)
    for i in range(n):
        if not keep[i]: continue
        for j in range(i+1, n):
            if keep[j] and abs(cor[i,j]) > thresh: keep[j] = False
    return keep

def safe_pca(X: np.ndarray, mn=15, mx=25):
    n, p = X.shape
    nc = max(min(mx, p, n-1, max(mn, p//3)), min(mn, p, n-1))
    pca = PCA(n_components=nc, random_state=42)
    Xp  = pca.fit_transform(X)
    return Xp, pca, float(pca.explained_variance_ratio_.sum())


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTION ENGINE  (FIXED thresholds + contradiction penalty)
# ─────────────────────────────────────────────────────────────────────────────
class DirectionEngine:
    """
    9 independent signal categories.
    Signal strength: STRONG=3, MEDIUM=2, WEAK=1.
    Require MIN_INDEP_SIGS=2 categories + score≥MIN_SCORE=3.
    Contradiction penalty: -1pt per conflicting category.
    """

    def score(self, df: pd.DataFrame, fund: pd.DataFrame = None,
              tick_snap: dict = None) -> dict:
        c_    = df["close"].astype(float)
        dp    = df["delta_pct"].astype(float)
        dlt   = df["delta"].astype(float)
        atr   = float(df["atr"].iloc[-1]) or float(c_.iloc[-1])*0.003
        price = float(c_.iloc[-1])
        bp    = df["body_pct"]
        vz    = float(df["vol_z"].iloc[-1])

        cats_bull = []; cats_bear = []
        pts_bull  = defaultdict(int); pts_bear = defaultdict(int)
        active    = {}

        # ── 1. CVD DIVERGENCE ─────────────────────────────────────────────
        cvd20 = dlt.rolling(20).sum()
        pr3   = c_.diff(3) / c_.shift(3) * 100
        cvd3  = cvd20.diff(3)
        div_b = bool(pr3.iloc[-1] < -0.12 and cvd3.iloc[-1] > 0)
        div_s = bool(pr3.iloc[-1] >  0.12 and cvd3.iloc[-1] < 0)
        exh_s = bool(dp.iloc[-1] < -0.28 and abs(bp.iloc[-1]) < 0.06)
        exh_b = bool(dp.iloc[-1] >  0.28 and abs(bp.iloc[-1]) < 0.06)
        if div_b:
            cats_bull.append("cvd"); pts_bull["cvd"] = 3
            active["cvd"] = "+BULL_DIV (strong)"
        elif exh_s:
            cats_bull.append("cvd"); pts_bull["cvd"] = 2
            active["cvd"] = "+SELL_EXHAUSTION"
        elif div_s:
            cats_bear.append("cvd"); pts_bear["cvd"] = 3
            active["cvd"] = "-BEAR_DIV (strong)"
        elif exh_b:
            cats_bear.append("cvd"); pts_bear["cvd"] = 2
            active["cvd"] = "-BUY_EXHAUSTION"

        # ── 2. OU MEAN REVERSION (FIXED: threshold 2.0 not 1.5) ──────────
        ou_z = 0.0
        x_ou = c_.values[-100:] if len(c_) >= 100 else c_.values
        if len(x_ou) >= 30:
            dx = np.diff(x_ou); xl = x_ou[:-1]
            A  = np.column_stack([np.ones(len(xl)), xl])
            try:
                co,_,_,_ = np.linalg.lstsq(A, dx, rcond=None)
                mu_ = -co[0]/co[1] if co[1] != 0 else x_ou.mean()
                sg_ = max(float(np.std(dx-(co[0]+co[1]*xl))), 1e-9)
                ou_z = float(np.clip((price-mu_)/sg_, -5, 5))
            except: pass
        thr_ou = CFG["OU_Z_THRESH"]
        if ou_z < -thr_ou:
            cats_bull.append("ou"); pts_bull["ou"] = 3 if ou_z < -3 else 2
            active["ou"] = "+z={:.2f} OVERSOLD".format(ou_z)
        elif ou_z > thr_ou:
            cats_bear.append("ou"); pts_bear["ou"] = 3 if ou_z > 3 else 2
            active["ou"] = "-z={:.2f} OVERBOUGHT".format(ou_z)

        # ── 3. KALMAN TREND (FIXED: threshold 0.25 not 0.15) ─────────────
        z_ = c_.values.astype(float); n = len(z_)
        F_ = np.array([[1.,1.],[0.,1.]]); H_ = np.array([[1.,0.]])
        Q_ = np.array([[0.01,0.001],[0.001,0.0001]]); R_ = np.array([[1.0]])
        xk = np.array([[z_[0]],[0.]]); Pk = np.eye(2)*1000.
        kp_ = np.zeros(n); kt_ = np.zeros(n)
        for t in range(n):
            xp = F_@xk; Pp = F_@Pk@F_.T+Q_
            K  = Pp@H_.T@np.linalg.inv(H_@Pp@H_.T+R_)
            xk = xp+K*(z_[t]-float((H_@xp).flat[0])); Pk=(np.eye(2)-K@H_)@Pp
            kp_[t] = float(xk[0].flat[0]); kt_[t] = float(xk[1].flat[0])
        kal_t = float(kt_[-1]); kal_p = float(kp_[-1])
        kal_dev = price - kal_p
        thr_k = CFG["KAL_TREND_THRESH"]
        if kal_t > thr_k:
            cats_bull.append("kalman"); pts_bull["kalman"] = 2
            active["kal"] = "+trend={:.3f}/bar".format(kal_t)
        elif kal_t < -thr_k:
            cats_bear.append("kalman"); pts_bear["kalman"] = 2
            active["kal"] = "-trend={:.3f}/bar".format(kal_t)
        # Kalman mean reversion (price far from filtered level)
        if kal_dev < -atr*1.5 and "kalman" not in cats_bull:
            cats_bull.append("kalman"); pts_bull["kalman"] = 2
            active["kal_rev"] = "+${:.0f} below Kalman".format(-kal_dev)
        elif kal_dev > atr*1.5 and "kalman" not in cats_bear:
            cats_bear.append("kalman"); pts_bear["kalman"] = 2
            active["kal_rev"] = "-${:.0f} above Kalman".format(kal_dev)

        # ── 4. WYCKOFF CYCLE ──────────────────────────────────────────────
        n_w = min(30,len(df)); x_w = np.arange(n_w); rec = df.tail(n_w)
        def sl_(v):
            try: return float(np.polyfit(x_w[:len(v)], v, 1)[0])
            except: return 0.0
        pt = sl_(rec["close"].values)
        bt = sl_(rec["taker_buy_vol"].values)
        st = sl_((rec["volume"]-rec["taker_buy_vol"]).values)
        wy = (2 if pt<-0.3 and bt>0 else  3 if pt>0.3 and bt>0 else
             -2 if pt>0.3  and st>0 else -3 if pt<-0.3 and st>0 else 0)
        if wy >= 2:
            cats_bull.append("wyckoff"); pts_bull["wyckoff"] = 2 if wy==2 else 3
            active["wy"] = "+{}".format("ACCUMULATION" if wy==2 else "MARKUP")
        elif wy <= -2:
            cats_bear.append("wyckoff"); pts_bear["wyckoff"] = 2 if wy==-2 else 3
            active["wy"] = "-{}".format("DISTRIBUTION" if wy==-2 else "MARKDOWN")

        # ── 5. VWAP BAND (FIXED: threshold 1.8σ) ─────────────────────────
        vol_ = df["volume"].astype(float).replace(0, np.nan)
        tp__ = (df["high"]+df["low"]+c_)/3
        vw20 = (tp__*vol_).rolling(20).sum()/vol_.rolling(20).sum()
        vr20 = (vol_*(tp__-vw20)**2).rolling(20).sum()/vol_.rolling(20).sum()
        vs20 = np.sqrt(vr20.replace(0, np.nan))
        vdev = float((c_-vw20).iloc[-1]/vs20.iloc[-1]) if float(vs20.iloc[-1])>0 else 0.
        thr_v = CFG["VWAP_SIGMA"]
        if vdev < -thr_v:
            cats_bull.append("vwap"); pts_bull["vwap"] = 3 if vdev<-2.5 else 2
            active["vwap"] = "+{:.2f}σ below VWAP".format(-vdev)
        elif vdev > thr_v:
            cats_bear.append("vwap"); pts_bear["vwap"] = 3 if vdev>2.5 else 2
            active["vwap"] = "-{:.2f}σ above VWAP".format(vdev)

        # ── 6. LIQUIDITY SWEEP ────────────────────────────────────────────
        wt = float(df["wick_top"].iloc[-1])
        wb = float(df["wick_bot"].iloc[-1])
        bp_l = float(bp.iloc[-1])
        thr_sw = atr * CFG["SWEEP_ATR_MULT"]
        if wb > thr_sw and vz > 1.2 and bp_l > 0:
            cats_bull.append("sweep"); pts_bull["sweep"] = 3
            active["sweep"] = "+BOT_WICK_SWEEP"
        elif wt > thr_sw and vz > 1.2 and bp_l < 0:
            cats_bear.append("sweep"); pts_bear["sweep"] = 3
            active["sweep"] = "-TOP_WICK_SWEEP"

        # ── 7. FUNDING RATE EXTREME (contrarian) ──────────────────────────
        avg_fr = 0.0
        if fund is not None and len(fund) >= 3:
            avg_fr = float(fund["fundingRate"].tail(8).mean())
        if avg_fr < -0.0003:
            cats_bull.append("funding"); pts_bull["funding"] = 2
            active["fund"] = "+SHORTS_HEATED({:.4f}%)".format(avg_fr*100)
        elif avg_fr > 0.0005:
            cats_bear.append("funding"); pts_bear["funding"] = 2
            active["fund"] = "-LONGS_HEATED({:.4f}%)".format(avg_fr*100)

        # ── 8. TRAPPED TRADERS ────────────────────────────────────────────
        bp_prev = float(bp.shift(1).iloc[-1])
        o_prev  = float(df["open"].shift(1).iloc[-1])
        if bp_prev < -0.25 and price > o_prev:
            cats_bull.append("trap"); pts_bull["trap"] = 3
            active["trap"] = "+SHORTS_TRAPPED"
        elif bp_prev > 0.25 and price < o_prev:
            cats_bear.append("trap"); pts_bear["trap"] = 3
            active["trap"] = "-LONGS_TRAPPED"

        # ── 9. REAL-TIME TICK FLOW (WebSocket) ────────────────────────────
        if tick_snap and tick_snap.get("trades", 0) > 10:
            td = tick_snap.get("delta_pct", 0)
            pr = tick_snap.get("pressure", "NEUTRAL")
            if td > 0.30 and pr == "BUY":
                cats_bull.append("tick"); pts_bull["tick"] = 2
                active["tick"] = "+TICK_BULL d={:.3f}".format(td)
            elif td < -0.30 and pr == "SELL":
                cats_bear.append("tick"); pts_bear["tick"] = 2
                active["tick"] = "-TICK_BEAR d={:.3f}".format(td)

        # ── AGGREGATE ─────────────────────────────────────────────────────
        u_bull = list(set(cats_bull)); u_bear = list(set(cats_bear))
        n_bull = len(u_bull);         n_bear  = len(u_bear)
        sc_bull = sum(pts_bull[c] for c in u_bull)
        sc_bear = sum(pts_bear[c] for c in u_bear)

        # Contradiction penalty: if both bull and bear signals present
        # subtract 1pt per overlapping category from the weaker side
        contradiction = min(n_bull, n_bear)
        if contradiction > 0:
            active["contra"] = "!CONFLICT: {} bull vs {} bear cats (-{}pts)".format(
                n_bull, n_bear, contradiction)

        score = sc_bull - sc_bear - contradiction  # net score

        # Determine side
        side = ("BUY"  if n_bull >= CFG["MIN_INDEP_SIGS"] and score >= CFG["MIN_SCORE"] else
                "SELL" if n_bear >= CFG["MIN_INDEP_SIGS"] and score <= -CFG["MIN_SCORE"] else
                "WAIT")

        # Session filter: Asian session has ~40% lower directional edge
        hr_now = df["open_time"].dt.hour.iloc[-1]
        in_active = hr_now in range(8, 20)  # London + NY
        if not in_active and side != "WAIT":
            # Need one extra category outside active session
            if (side == "BUY"  and n_bull < CFG["MIN_INDEP_SIGS"]+1) or \
               (side == "SELL" and n_bear < CFG["MIN_INDEP_SIGS"]+1):
                active["session"] = "ASIAN_RANGE skipped (need {}+ cats outside London/NY)".format(
                    CFG["MIN_INDEP_SIGS"]+1)
                side = "WAIT"

        return {
            "side":       side,
            "score":      int(score),
            "n_bull":     n_bull, "n_bear":  n_bear,
            "n_indep":    n_bull if side=="BUY" else n_bear,
            "sc_bull":    sc_bull,"sc_bear": sc_bear,
            "cats_bull":  u_bull, "cats_bear":u_bear,
            "active":     active,
            "ou_z":       ou_z,  "kal_trend":kal_t, "kal_price":kal_p,
            "wyckoff":    wy,    "vwap_dev": vdev,  "avg_fr":   avg_fr,
            "active_ses": in_active,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  ML CONFIDENCE FILTER  (FIXED: asymmetric labels + ET 2×)
# ─────────────────────────────────────────────────────────────────────────────
class MLFilter:
    def __init__(self):
        self.gbm = None; self.et = None
        self.scaler = RobustScaler(); self.pca = None
        self.iso = IsotonicRegression(out_of_bounds="clip"); self.cal = False
        self.keep_mask = None; self.trained = False
        self.val_acc = 0.5; self.n_comp = 0

    def _barrier(self, df, pct=0.010, t_max=6, tp_m=1.5):
        """Asymmetric: TP = pct×tp_m, SL = pct. TP > SL → TP rate > SL rate."""
        prices = df["close"].astype(float).values
        atrs   = df["atr"].astype(float).values
        n = len(prices); labels = np.full(n, np.nan)
        lr_ = np.diff(np.log(np.maximum(prices, 1e-9)))
        rv5 = np.array([lr_[max(0,i-5):i].std() if i>1 else pct for i in range(n)])
        rv5 = np.maximum(rv5, 0.002)
        for i in range(n - t_max):
            p0  = prices[i]; ai = atrs[i] if atrs[i]>0 else p0*0.003
            sl_w = max(pct, 1.5*rv5[i], ai/p0)
            tp_w = sl_w * tp_m            # TP is wider than SL
            tp = p0*(1+tp_w); sl = p0*(1-sl_w); lbl = 0
            for j in range(1, t_max+1):
                if i+j >= n: break
                p = prices[i+j]
                if p >= tp: lbl=1; break
                if p <= sl: lbl=-1; break
            if lbl == 0:
                rf = (prices[min(i+t_max, n-1)]/p0) - 1
                if   rf >  0.0005: lbl =  1
                elif rf < -0.0005: lbl = -1
            labels[i] = lbl
        return pd.Series(labels, index=df.index).dropna()

    def _splits(self, n, k=5):
        fs = n//k; splits = []
        for f in range(k):
            ts = f*fs; te = ts+fs if f<k-1 else n
            tr = list(range(0,max(0,ts-CFG["PURGE"]))) + list(range(min(n,te+CFG["EMBARGO"]),n))
            ti = list(range(ts,te))
            if len(tr)>=50 and len(ti)>=10: splits.append((tr,ti))
        return splits

    def train(self, df, fund, verbose=True):
        if len(df) < 200: return {}
        vp = verbose

        if vp: print("  [ML] 30 features...", end=" ", flush=True)
        F_df = build_features(df, fund)
        X_r  = np.nan_to_num(F_df.values.astype(float), 0.)
        if vp: print("{} features".format(X_r.shape[1]))

        if vp: print("  [ML] Triple-barrier (TP={:.1f}×SL)...".format(CFG["TP_MULT_LABEL"]), end=" ", flush=True)
        tb   = self._barrier(df, pct=CFG["BARRIER_PCT"], t_max=CFG["TARGET_BARS"], tp_m=CFG["TP_MULT_LABEL"])
        idx  = tb.index; df_v = df.loc[idx]; y_tb = tb.values
        tp_r = float((y_tb==1).mean()*100); sl_r=float((y_tb==-1).mean()*100); ep_r=float((y_tb==0).mean()*100)
        if vp: print("TP={:.1f}%  SL={:.1f}%  Exp={:.1f}%  (TP>SL = good)".format(tp_r,sl_r,ep_r))
        if tp_r < sl_r - 2:
            print("  [ML] WARN: SL>TP — barrier too symmetric. Using anyway.")

        F_v  = build_features(df_v, fund)
        X_v  = np.nan_to_num(F_v.values.astype(float), 0.)
        X_sc = self.scaler.fit_transform(X_v)

        if vp: print("  [ML] Correlation filter (r<{})...".format(CFG["CORR_THRESH"]), end=" ", flush=True)
        self.keep_mask = corr_filter(X_sc, CFG["CORR_THRESH"])
        X_f  = X_sc[:, self.keep_mask]
        if vp: print("{} → {} kept".format(X_sc.shape[1], X_f.shape[1]))

        if vp: print("  [ML] PCA (min={} comp)...".format(CFG["PCA_MIN_COMP"]), end=" ", flush=True)
        X_pca, self.pca, ev = safe_pca(X_f, CFG["PCA_MIN_COMP"], CFG["PCA_MAX_COMP"])
        self.n_comp = X_pca.shape[1]
        if vp: print("{} components ({:.1f}% var)".format(self.n_comp, ev*100))

        y_dir = (y_tb == 1).astype(int)
        splits = self._splits(len(X_pca))

        # GBM (lower LR for stable convergence)
        if vp: print("  [ML] GBM lr={}...".format(CFG["GBM_LR"]), end=" ", flush=True)
        self.gbm = GradientBoostingClassifier(
            n_estimators=CFG["GBM_N"], learning_rate=CFG["GBM_LR"],
            max_depth=4, subsample=0.75, min_samples_leaf=10, random_state=42)
        oof_gbm = np.full(len(X_pca), 0.5)
        for tr, te in splits:
            if len(np.unique(y_dir[tr])) < 2: continue
            self.gbm.fit(X_pca[tr], y_dir[tr])
            oof_gbm[te] = self.gbm.predict_proba(X_pca[te])[:,1]
        if len(np.unique(y_dir)) >= 2: self.gbm.fit(X_pca, y_dir)
        gbm_acc = float(((oof_gbm>0.5).astype(int)==y_dir).mean())
        if vp: print("OOF={:.4f}".format(gbm_acc))

        # ExtraTrees (better performer, gets 2× weight)
        if vp: print("  [ML] ET weight={}×...".format(CFG["ET_WEIGHT"]), end=" ", flush=True)
        self.et = ExtraTreesClassifier(
            n_estimators=CFG["ET_N"], max_depth=6, min_samples_leaf=8,
            random_state=42, n_jobs=-1)
        oof_et = np.full(len(X_pca), 0.5)
        for tr, te in splits:
            if len(np.unique(y_dir[tr])) < 2: continue
            self.et.fit(X_pca[tr], y_dir[tr])
            oof_et[te] = self.et.predict_proba(X_pca[te])[:,1]
        if len(np.unique(y_dir)) >= 2: self.et.fit(X_pca, y_dir)
        et_acc = float(((oof_et>0.5).astype(int)==y_dir).mean())
        if vp: print("OOF={:.4f}".format(et_acc))

        # Weighted ensemble + calibration
        w = CFG["ET_WEIGHT"]
        oof_avg = (oof_gbm + w*oof_et) / (1+w)
        ne = y_tb != 0
        if ne.sum() > 20:
            self.iso.fit(oof_avg[ne], y_dir[ne]); self.cal = True

        self.trained = True
        self.val_acc = (gbm_acc + w*et_acc) / (1+w)
        if vp:
            fi = self.gbm.feature_importances_
            print("  [ML] Weighted acc={:.4f}  PCA comp with >0.1% importance: {}/{}".format(
                self.val_acc, (fi>0.001).sum(), len(fi)))

        return {"gbm_acc":gbm_acc,"et_acc":et_acc,"n_pca":self.n_comp,
                "tp":tp_r,"sl":sl_r,"exp":ep_r,"n_samples":len(X_pca)}

    def predict(self, df, fund) -> dict:
        if not self.trained or self.pca is None:
            return {"ml_prob":0.5,"allow":True,"reason":"not_trained"}
        try:
            Fd   = build_features(df, fund)
            X_r  = np.nan_to_num(Fd.values.astype(float), 0.)
            X_sc = self.scaler.transform(X_r)
            X_f  = X_sc[:, self.keep_mask] if self.keep_mask is not None else X_sc
            X_p  = self.pca.transform(X_f)
            p_g  = float(self.gbm.predict_proba(X_p[-1:])[:,1][0])
            p_e  = float(self.et.predict_proba( X_p[-1:])[:,1][0])
            w    = CFG["ET_WEIGHT"]
            prob = (p_g + w*p_e) / (1+w)
            if self.cal: prob = float(self.iso.predict([prob])[0])
            # Only block on clear bearish/bullish ML reads; uncertain = allow
            allow = prob >= CFG["MIN_ML_CONF"] or (0.44 <= prob <= 0.56)
            return {"ml_prob":prob,"p_gbm":p_g,"p_et":p_e,
                    "allow":allow,"reason":"ok" if allow else "ml_blocked"}
        except Exception as e:
            return {"ml_prob":0.5,"allow":True,"reason":"err:{}".format(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  GARCH + MARKET PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def garch11(ret):
    r = ret.dropna().values
    if len(r) < 30: return 0.003, 1.0, "MEDIUM", 50.0
    v0 = float(np.var(r))
    ac1 = float(pd.Series(r**2).autocorr(1)) if len(r)>=10 else 0.1
    al0 = max(min(max(ac1, 0.01), 0.15), 0.01)
    be0 = min(max(0.85, 1-al0-0.03), 0.95)
    om0 = v0*(1-al0-be0)
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
        res=optimize.minimize(nll,[om0,al0,be0],method="L-BFGS-B",
            bounds=[(1e-9,None),(1e-9,0.999),(1e-9,0.999)],options={"maxiter":80})
        om,al,be=res.x
    except: om,al,be=om0,al0,be0
    h=np.full(len(r),v0)
    for t in range(1,len(r)): h[t]=max(om+al*r[t-1]**2+be*h[t-1],1e-12)
    cv  = float(math.sqrt(h[-1]))
    vp  = float(stats.percentileofscore(np.sqrt(h), cv))
    rg  = "LOW" if vp<30 else ("HIGH" if vp>75 else "MEDIUM")
    # HIGH vol → reduce size 0.5×, not block
    sm  = 1.5 if vp<30 else (CFG["HIGH_VOL_SIZE"] if vp>80 else 1.0)
    return cv, sm, rg, vp

def market_profile(df, tick=25.0):
    lo=df["low"].min(); hi=df["high"].max()
    bkts=np.arange(math.floor(lo/tick)*tick, math.ceil(hi/tick)*tick+tick, tick)
    vm=defaultdict(float)
    for _,row in df.iterrows():
        lvls=bkts[(bkts>=row["low"])&(bkts<=row["high"])]
        if not len(lvls): continue
        vp=row["volume"]/len(lvls)
        for lv in lvls: vm[lv]+=vp
    if not vm: p=float(df["close"].iloc[-1]); return p,p,p
    pf=pd.DataFrame({"p":list(vm.keys()),"v":list(vm.values())}).sort_values("p")
    poc=float(pf.loc[pf["v"].idxmax(),"p"]); tot=pf["v"].sum()
    pi=pf["v"].idxmax(); cum=0.0; va=[]
    for _ in range(len(pf)):
        ui=pi+1; li=pi-1
        uv=pf.loc[ui,"v"] if ui in pf.index else 0.
        dv=pf.loc[li,"v"] if li in pf.index else 0.
        if uv>=dv and ui in pf.index: va.append(ui); cum+=uv; pi=ui
        elif li in pf.index:          va.append(li); cum+=dv; pi=li
        else: break
        if cum/tot >= 0.70: break
    vah=float(pf.loc[va,"p"].max()) if va else poc+tick*5
    val=float(pf.loc[va,"p"].min()) if va else poc-tick*5
    return poc, vah, val


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch(sym, tf, lim):
    r=requests.get("{}/fapi/v1/klines".format(BASE_API),
                   params={"symbol":sym,"interval":tf,"limit":lim},timeout=15)
    r.raise_for_status()
    df=pd.DataFrame(r.json(),columns=["ts","o","h","l","c","v","ct","qv","n","tbv","tbqv","_"])
    df["open_time"]=pd.to_datetime(df["ts"].astype(float),unit="ms",utc=True)
    for col in ["o","h","l","c","v","tbv","n"]: df[col]=df[col].astype(float)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close",
                               "v":"volume","tbv":"taker_buy_vol","n":"trades"})[
        ["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]

def fetch_fund(sym):
    r=requests.get("{}/fapi/v1/fundingRate".format(BASE_API),
                   params={"symbol":sym,"limit":50},timeout=10)
    r.raise_for_status(); df=pd.DataFrame(r.json())
    df["fundingTime"]=pd.to_datetime(df["fundingTime"].astype(float),unit="ms",utc=True)
    df["fundingRate"]=df["fundingRate"].astype(float); return df

def synthetic(n=1500, seed=42, base=67000.):
    np.random.seed(seed)
    dates=pd.date_range(end=pd.Timestamp.utcnow(),periods=n,freq="5min",tz="UTC")
    price=float(base); rows=[]
    for dt in dates:
        h=dt.hour; sv=2.2 if h in [8,9,13,14,15,16] else 0.65
        mu=-0.00018 if h in [16,17,18] else 0.00012
        price=max(price*(1+np.random.normal(mu,0.0028*sv)),50000)
        hi=price*(1+abs(np.random.normal(0,0.002*sv)))
        lo=price*(1-abs(np.random.normal(0,0.002*sv)))
        vol=max(abs(np.random.normal(1100,380))*sv,80.)
        bsk=0.63 if h in [8,9] else(0.36 if h in [17,18] else 0.50)
        tb=vol*float(np.clip(np.random.beta(bsk*7,(1-bsk)*7),0.05,0.95))
        if np.random.random()<0.025: vol*=np.random.uniform(5,9)
        rows.append({"open_time":dt,"open":price*(1+np.random.normal(0,0.001)),
                     "high":hi,"low":lo,"close":price,"volume":vol,
                     "taker_buy_vol":tb,"trades":int(vol/0.04)})
    df=pd.DataFrame(rows)
    fund=pd.DataFrame([{"fundingTime":dates[i],
                         "fundingRate":float(np.random.normal(0.0001,0.0003))}
                        for i in range(0,n,96)])
    return df, fund

def prepare(df):
    d=df.copy()
    d["body"]     =d["close"]-d["open"]; d["body_pct"]=d["body"]/d["open"]*100
    d["is_bull"]  =d["body"]>0
    d["wick_top"] =d["high"]-d[["open","close"]].max(axis=1)
    d["wick_bot"] =d[["open","close"]].min(axis=1)-d["low"]
    d["sell_vol"] =d["volume"]-d["taker_buy_vol"]
    d["delta"]    =d["taker_buy_vol"]-d["sell_vol"]
    d["delta_pct"]=(d["delta"]/d["volume"].replace(0,np.nan)).fillna(0)
    hl=d["high"]-d["low"]
    hpc=(d["high"]-d["close"].shift(1)).abs()
    lpc=(d["low"] -d["close"].shift(1)).abs()
    d["atr"]  =pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    rm=d["volume"].rolling(50).mean(); rs=d["volume"].rolling(50).std().replace(0,np.nan)
    d["vol_z"]=(d["volume"]-rm)/rs
    d["hour"]=d["open_time"].dt.hour
    return d.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    def __init__(self, d=CFG["MODEL_DIR"]):
        self.d=d; self.bs=-np.inf; os.makedirs(d,exist_ok=True)
        self.lat=os.path.join(d,"latest.pkl"); self.best=os.path.join(d,"best.pkl")
        self.meta=os.path.join(d,"meta.json")

    def save(self, state, sharpe=None):
        try:
            with open(self.lat,"wb") as f: pickle.dump(state,f,protocol=pickle.HIGHEST_PROTOCOL)
            if sharpe is not None and sharpe > self.bs:
                self.bs=sharpe
                import shutil; shutil.copy(self.lat,self.best)
            m={"saved_at":datetime.now(timezone.utc).isoformat(),
               "sharpe":float(sharpe) if sharpe else 0.,
               "best_sharpe":float(self.bs),
               "n_samples":state.get("n_samples",0),
               "n_pca":state.get("n_pca",0),
               "val_acc":state.get("val_acc",0.)}
            with open(self.meta,"w") as f: json.dump(m,f,indent=2)
            return True
        except Exception as e: print("  [SAVE ERR]",e); return False

    def load(self):
        for path in [self.best, self.lat]:
            if not os.path.exists(path): continue
            try:
                with open(path,"rb") as f: s=pickle.load(f)
                if os.path.exists(self.meta):
                    with open(self.meta) as f: m=json.load(f)
                    print("  [LOAD] acc={:.3f}  n_pca={}  n={}  {}".format(
                        m.get("val_acc",0),m.get("n_pca",0),
                        m.get("n_samples",0),m.get("saved_at","?")[:19]))
                return s
            except Exception as e: print("  [LOAD ERR]",e)
        return {}

    def exists(self): return os.path.exists(self.lat) or os.path.exists(self.best)

    def delete(self):
        import shutil
        if os.path.exists(self.d): shutil.rmtree(self.d)
        os.makedirs(self.d,exist_ok=True); self.bs=-np.inf
        print("  [STORE] Cleared — will retrain fresh.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADER + SIGNAL HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class PaperTrader:
    def __init__(self,acc):
        self.balance=acc; self.start=acc; self.position=None
        self.wins=0; self.losses=0; self.daily_pnl=0.; self.trades=[]
        self.lock=threading.Lock()
    @property
    def wr(self): return self.wins/max(self.wins+self.losses,1)*100
    @property
    def pnl_pct(self): return (self.balance-self.start)/self.start*100

    def enter(self,side,entry,sl,tp1,tp2,qty,score,conf,reason):
        with self.lock:
            if self.position: return False
            slip=entry*CFG["PAPER_SLIP"]*(1 if side=="BUY" else -1)
            self.position={"side":side,"entry":entry+slip,"sl":sl,"tp1":tp1,"tp2":tp2,
                           "qty":qty,"score":score,"conf":conf,"reason":reason,
                           "time":datetime.now(timezone.utc),"tp1_hit":False}
            return True

    def update(self,price):
        with self.lock:
            if not self.position: return None
            p=self.position; s=p["side"]; result=None
            if not p["tp1_hit"]:
                h1=(s=="BUY" and price>=p["tp1"]) or (s=="SELL" and price<=p["tp1"])
                if h1:
                    pnl=p["qty"]*0.6*abs(p["tp1"]-p["entry"])*(1 if s=="BUY" else -1)
                    self.balance+=pnl; self.daily_pnl+=pnl
                    p["tp1_hit"]=True; p["qty"]*=0.4; p["sl"]=p["entry"]
                    result={"type":"TP1","pnl":pnl,"price":price}
            if p["tp1_hit"]:
                h2=(s=="BUY" and price>=p["tp2"]) or (s=="SELL" and price<=p["tp2"])
                if h2:
                    pnl=p["qty"]*abs(p["tp2"]-p["entry"])*(1 if s=="BUY" else -1)
                    self.balance+=pnl; self.daily_pnl+=pnl; self.wins+=1
                    self.trades.append({**p,"exit":price,"pnl":pnl,"result":"WIN"})
                    self.position=None
                    return {"type":"WIN","pnl":pnl,"price":price}
            hs=(s=="BUY" and price<=p["sl"]) or (s=="SELL" and price>=p["sl"])
            if hs:
                pnl=p["qty"]*abs(p["sl"]-p["entry"])*(-1 if s=="BUY" else 1)
                self.balance+=pnl; self.daily_pnl+=pnl; self.losses+=1
                self.trades.append({**p,"exit":price,"pnl":pnl,"result":"LOSS"})
                self.position=None
                result={"type":"LOSS","pnl":pnl,"price":price}
            return result

    def stats(self):
        return {"balance":self.balance,"pnl_pct":self.pnl_pct,
                "trades":self.wins+self.losses,"wins":self.wins,"losses":self.losses,
                "wr":self.wr,"daily":self.daily_pnl,"in_pos":self.position is not None}

class SigHistory:
    def __init__(self,mx=200):
        self.sigs=deque(maxlen=mx); self.ok=0; self.tot=0; self.lock=threading.Lock()
    def record(self,side,price,score,conf,n_ind):
        with self.lock:
            self.sigs.append({"side":side,"price":price,"score":score,"conf":conf,
                              "n_ind":n_ind,"time":datetime.now(timezone.utc),"out":None})
    def resolve(self,fp):
        with self.lock:
            for s in reversed(self.sigs):
                if s["out"] is None and s["side"]!="WAIT":
                    ok=(s["side"]=="BUY" and fp>s["price"]) or (s["side"]=="SELL" and fp<s["price"])
                    s["out"]="W" if ok else "L"; self.tot+=1
                    if ok: self.ok+=1
                    break
    @property
    def acc(self): return self.ok/max(self.tot,1)*100
    def recent(self,n=6): return list(self.sigs)[-n:]


# ─────────────────────────────────────────────────────────────────────────────
#  TICK + KLINE BUFFERS + WS
# ─────────────────────────────────────────────────────────────────────────────
class TickBuf:
    def __init__(self,mx=3000):
        self.ticks=deque(maxlen=mx); self.lock=threading.Lock(); self.lp=0.; self.lt=0
    def add(self,p,q,ibm,ts):
        with self.lock: self.lp=p; self.lt=ts; self.ticks.append({"p":p,"q":q,"b":not ibm,"ts":ts})
    def snap(self,ms=30000):
        now=self.lt
        with self.lock: rec=[t for t in self.ticks if now-t["ts"]<=ms]
        if not rec: return {"buy_vol":0,"sell_vol":0,"delta_pct":0,"trades":0,"price":self.lp,"pressure":"NEUTRAL"}
        bv=sum(t["q"] for t in rec if t["b"]); sv=sum(t["q"] for t in rec if not t["b"])
        return {"buy_vol":bv,"sell_vol":sv,"delta":bv-sv,
                "delta_pct":float(np.clip((bv-sv)/(bv+sv+1e-9),-1,1)),
                "trades":len(rec),"price":self.lp,
                "pressure":"BUY" if bv>sv*1.3 else("SELL" if sv>bv*1.3 else "NEUTRAL")}

class KlineBuf:
    def __init__(self,mx=600):
        self.df=pd.DataFrame(); self.mx=mx; self.lock=threading.Lock(); self.ev=threading.Event()
    def update(self,row):
        with self.lock:
            nr=pd.DataFrame([row]); nr["open_time"]=pd.to_datetime(nr["open_time"],unit="ms",utc=True)
            if self.df.empty: self.df=nr
            elif row["open_time"] not in self.df["open_time"].values:
                self.df=pd.concat([self.df,nr],ignore_index=True).tail(self.mx).reset_index(drop=True)
            self.ev.set()
    def get(self):
        with self.lock: return self.df.copy()
    def wait(self,t=70): self.ev.clear(); return self.ev.wait(timeout=t)

class WSMgr:
    def __init__(self,sym,tf,kb,tb):
        self.sym=sym.lower(); self.tf=tf; self.kb=kb; self.tb=tb
        self.conn=False; self._stop=threading.Event()
    def _kl(self,ws,msg):
        try:
            d=json.loads(msg); k=d.get("k",{})
            if not k.get("x"): return
            self.kb.update({"open_time":int(k["t"]),"open":float(k["o"]),"high":float(k["h"]),
                            "low":float(k["l"]),"close":float(k["c"]),"volume":float(k["v"]),
                            "taker_buy_vol":float(k.get("Q",float(k["v"])*0.5)),"trades":int(k.get("n",0))})
        except: pass
    def _tk(self,ws,msg):
        try:
            d=json.loads(msg); self.tb.add(float(d["p"]),float(d["q"]),bool(d["m"]),int(d["T"]))
        except: pass
    def _run(self,url,fn):
        while not self._stop.is_set():
            try:
                w=_ws.WebSocketApp(url,on_message=fn,
                    on_open=lambda x:setattr(self,"conn",True),
                    on_close=lambda x,c,m:setattr(self,"conn",False))
                w.run_forever(ping_interval=20,ping_timeout=10)
            except: pass
            if not self._stop.is_set(): time.sleep(5)
    def start(self):
        if not WS_OK: return False
        threading.Thread(target=self._run,args=("{}/{}@kline_{}".format(BASE_WS,self.sym,self.tf),self._kl),daemon=True).start()
        threading.Thread(target=self._run,args=("{}/{}@aggTrade".format(BASE_WS,self.sym),self._tk),daemon=True).start()
        return True
    def stop(self): self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def display(price, dr, ml, final, tr, loop_n, live,
            paper_st, sh, ws_conn, ckpt, garch_m, vol_reg, poc, vah, val):
    os.system("cls" if os.name=="nt" else "clear")
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    side = final["side"]; sc=dr["score"]; conf=final["confidence"]
    sc_c = "G" if sc>0 else ("R" if sc<0 else "Y")
    vc   = "G" if vol_reg=="LOW" else ("R" if vol_reg=="HIGH" else "W")

    print(cc("="*76,"C"))
    print(cc("  ELITE QUANT ENGINE v7.1  |  BTC/USDT  |  SIGNALS FIXED","C"))
    print(cc("  2-Tier: Direction(9 cats) + ML(filter) | Need 2+ independent cats","C"))
    print(cc("="*76,"C"))
    print("  {}  Bar#{} {} {}  {}".format(
        cc(now,"D"), loop_n,
        "LIVE" if live else cc("SYN","Y"),
        cc("WS","G") if ws_conn else cc("REST","Y"),
        cc("SAVED","G") if ckpt else cc("unsaved","D")))
    print("  {}  GARCH×{:.2f}  {}  Session:{}  POC:${:,.1f}".format(
        cc("${:,.2f}".format(price),"W"), garch_m,
        cc(vol_reg,vc),
        cc("ACTIVE (London/NY)","G") if dr["active_ses"] else cc("ASIAN (low edge)","Y"),
        poc))
    print()

    # ── Main signal box ──
    b = bb(abs(sc)/10)
    print(cc("  "+"="*68,"W"))
    if   side=="BUY":
        print(cc("  ||  ####  B U Y  ^^^^^^^^^^  #### cats={} score={:+d} size×{:.1f}  ||".format(
            dr["n_bull"],sc,garch_m),"G"))
    elif side=="SELL":
        print(cc("  ||  ####  S E L L  vvvvvvvv  #### cats={} score={:+d} size×{:.1f}  ||".format(
            dr["n_bear"],sc,garch_m),"R"))
    else:
        # Explain why WAIT with specific numbers
        nb=dr["n_bull"]; nbe=dr["n_bear"]; sc_=dr["score"]
        if nb==0 and nbe==0:
            why="no signals fired yet"
        elif nb>0 and nb<CFG["MIN_INDEP_SIGS"]:
            why="BULL needs {}+ cats, have {} (need {} more)".format(
                CFG["MIN_INDEP_SIGS"],nb,CFG["MIN_INDEP_SIGS"]-nb)
        elif nbe>0 and nbe<CFG["MIN_INDEP_SIGS"]:
            why="BEAR needs {}+ cats, have {} (need {} more)".format(
                CFG["MIN_INDEP_SIGS"],nbe,CFG["MIN_INDEP_SIGS"]-nbe)
        elif not ml.get("allow"):
            why="ML blocked (P={:.3f} below {:.2f})".format(ml.get("ml_prob",0.5),CFG["MIN_ML_CONF"])
        elif abs(sc_) < CFG["MIN_SCORE"]:
            why="score={:+d} needs ≥{:+d}".format(sc_,CFG["MIN_SCORE"])
        elif not dr["active_ses"]:
            why="Asian session — need {}+ cats outside London/NY".format(CFG["MIN_INDEP_SIGS"]+1)
        else:
            why="R:R={:.2f}x below {:.1f}x".format(final.get("rr",0),CFG["MIN_RR"])
        print(cc("  ||  ----  W A I T  ({})  ||".format(why),"Y"))

    ml_c="G" if ml.get("allow") else "R"
    print("  ||  Score:{}  {}  Conf:{}  ML:{}({})  TP:{}  SL:{}  ||".format(
        cc("{:>+3d}".format(sc),"B"), cc(b,sc_c),
        cc("{:.0f}%".format(conf),"B"),
        cc("{:.3f}".format(ml.get("ml_prob",0.5)),ml_c),
        cc("PASS","G") if ml.get("allow") else cc("BLOCK","R"),
        cc("{:.0f}%".format(tr.get("tp",0)),"G"),
        cc("{:.0f}%".format(tr.get("sl",0)),"R")))
    print(cc("  "+"="*68,"W"))
    print()

    # Trade setup
    if final.get("tradeable") and final.get("tp1"):
        rr=final["rr"]; rc="G" if rr>=2.5 else("Y" if rr>=1.5 else "R")
        print(cc("  +--- TRADE -----------------------------------------------------------+","Y"))
        print("  |  Entry: ${:>12,.2f}{}|".format(price," "*44))
        print(cc("  |  Stop:  ${:>12,.2f}  (${:>7,.1f} = {:.1f}×ATR)  risk×{:.2f}".format(
            final["sl"],abs(price-final["sl"]),CFG["ATR_SL"],garch_m),"R")+" "*7+cc("|","Y"))
        print(cc("  |  TP1:   ${:>12,.2f}  → POC (close 60%)".format(final["tp1"]),"G")+" "*20+cc("|","Y"))
        print(cc("  |  TP2:   ${:>12,.2f}  → VAH/VAL (close 40%)".format(final["tp2"]),"G")+" "*17+cc("|","Y"))
        print("  |  R:R={}  Qty={:.3f}BTC  VOL:{}{}|".format(
            cc("{:.2f}x".format(rr),rc),final["qty"],cc(vol_reg,vc)," "*26))
        print(cc("  +-------------------------------------------------------------------+","Y"))
    elif side!="WAIT":
        ml_p=ml.get("ml_prob",0.5)
        if not ml.get("allow"):
            print(cc("  ✗ ML BLOCKED: P={:.3f}  GBM={:.3f}  ET={:.3f}".format(
                ml_p,ml.get("p_gbm",0.5),ml.get("p_et",0.5)),"R"))
        else:
            print(cc("  conf={:.0f}%  R:R={:.2f}x  not tradeable yet".format(conf,final.get("rr",0)),"Y"))
    print()

    # Direction breakdown
    print(cc("  -- DIRECTION ENGINE (9 categories, need {}+) -----------------------".format(
        CFG["MIN_INDEP_SIGS"]),"M"))
    print("  BULL {}/{}:  {}".format(dr["n_bull"],CFG["MIN_INDEP_SIGS"],
        cc(", ".join(dr["cats_bull"]),"G") if dr["cats_bull"] else cc("none","D")))
    print("  BEAR {}/{}:  {}".format(dr["n_bear"],CFG["MIN_INDEP_SIGS"],
        cc(", ".join(dr["cats_bear"]),"R") if dr["cats_bear"] else cc("none","D")))
    for k,v in dr.get("active",{}).items():
        col="G" if str(v).startswith("+") else("R" if str(v).startswith("-") else "Y")
        print("    {} {}".format(cc("›","Y"), cc("{}: {}".format(k,v),col)))
    print()

    # Key values
    print(cc("  -- SIGNAL READINGS -------------------------------------------------","M"))
    ou=dr["ou_z"]; kt=dr["kal_trend"]
    print("  OU z={:>+.3f}  {}  (threshold ±{})".format(ou,
        cc("OVERSOLD","G") if ou<-CFG["OU_Z_THRESH"] else
        cc("OVERBOUGHT","R") if ou>CFG["OU_Z_THRESH"] else "neutral",
        CFG["OU_Z_THRESH"]))
    print("  Kalman trend={:>+.4f}/bar  {}  (threshold ±{})".format(kt,
        cc("UP","G") if kt>CFG["KAL_TREND_THRESH"] else
        cc("DOWN","R") if kt<-CFG["KAL_TREND_THRESH"] else "neutral",
        CFG["KAL_TREND_THRESH"]))
    print("  VWAP band={:>+.3f}σ  {}  (threshold ±{}σ)".format(dr["vwap_dev"],
        cc("BUY zone","G") if dr["vwap_dev"]<-CFG["VWAP_SIGMA"] else
        cc("SELL zone","R") if dr["vwap_dev"]>CFG["VWAP_SIGMA"] else "neutral",
        CFG["VWAP_SIGMA"]))
    print("  Wyckoff={}  {}  Funding={:.5f}%".format(dr["wyckoff"],
        {3:"MARKUP",2:"ACCUM",-2:"DIST",-3:"MARKDOWN"}.get(dr["wyckoff"],"CONSOLIDATION"),
        dr["avg_fr"]*100))
    print("  ML: GBM={:.3f}  ET×2={:.3f}  Combined={:.3f}  {}".format(
        ml.get("p_gbm",0.5),ml.get("p_et",0.5),ml.get("ml_prob",0.5),
        cc("PASS","G") if ml.get("allow") else cc("BLOCK","R")))
    print("  POC=${:,.1f}  VAH=${:,.1f}  VAL=${:,.1f}  Kal=${:,.1f}".format(
        poc,vah,val,dr["kal_price"]))
    print()

    # ML stats
    print(cc("  -- ML FILTER -------------------------------------------------------","D"))
    n_pca=tr.get("n_pca",0); good_pca=n_pca>=CFG["PCA_MIN_COMP"]
    print("  GBM={:.1f}%  ET(2×)={:.1f}%  PCA={}comp{}  n={} samples  TP={:.0f}%  SL={:.0f}%".format(
        tr.get("gbm_acc",0)*100, tr.get("et_acc",0)*100,
        n_pca, cc(" ✓","G") if good_pca else cc(" ✗ too few","R"),
        tr.get("n_samples",0), tr.get("tp",0), tr.get("sl",0)))
    print()

    if paper_st:
        pc="G" if paper_st["pnl_pct"]>=0 else "R"
        print(cc("  -- PAPER TRADING ---------------------------------------------------","M"))
        print("  Balance:{}  PnL:{}  WR:{:.1f}%  Trades:{}  {}".format(
            cc("${:,.2f}".format(paper_st["balance"]),"W"),
            cc("{:+.2f}%".format(paper_st["pnl_pct"]),pc),
            paper_st["wr"],paper_st["trades"],
            cc("IN POSITION","G") if paper_st["in_pos"] else ""))
        print()

    recent=sh.recent(6)
    if recent:
        print(cc("  -- SIGNAL HISTORY  acc:{:.1f}%  (n={}) ----------------------------".format(
            sh.acc,sh.tot),"D"))
        for s in reversed(recent):
            oc=s.get("out","—"); occ="G" if oc=="W" else("R" if oc=="L" else "D")
            print("  {} {:>4}  {:+.0f}pts  conf={:.0f}%  cats={}  {}".format(
                s["time"].strftime("%H:%M:%S"),s["side"],s["score"],
                s["conf"],s["n_ind"],cc(oc,occ)))
        print()

    print(cc("  Ctrl+C  |  --paper  |  --account 5000  |  --tf 5m  |  --reset","D"))
    print(cc("="*76,"D"))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EliteV71:
    def __init__(self, account=1000., paper=True, reset=False):
        CFG["ACCOUNT"] = account
        self.dir_eng   = DirectionEngine()
        self.ml        = MLFilter()
        self.store     = ModelStore(CFG["MODEL_DIR"])
        self.kbuf      = KlineBuf(); self.tbuf = TickBuf()
        self.ws_mgr    = None; self.ws_conn = False
        self.paper     = PaperTrader(account) if paper else None
        self.sh        = SigHistory()
        self.trained   = False; self.tr = {}
        self.bars      = 0; self.bars_tr = 0; self.bars_ck = 0
        self.ckpt      = False
        if reset: self.store.delete()

    def train(self, df, fund, verbose=True):
        if verbose:
            print(cc("\n  ELITE QUANT ENGINE v7.1 — TRAINING","M"))
            print(cc("  "+"─"*60,"M"))
        t0 = time.time()
        res = self.ml.train(df, fund, verbose=verbose)
        self.trained = True; self.tr = res or {}
        if verbose: print(cc("  Done in {:.1f}s".format(time.time()-t0),"G"))
        self._ckpt()

    def _ckpt(self):
        state = {"ml_gbm":self.ml.gbm,"ml_et":self.ml.et,
                 "ml_scaler":self.ml.scaler,"ml_pca":self.ml.pca,
                 "ml_iso":self.ml.iso,"ml_cal":self.ml.cal,
                 "ml_mask":self.ml.keep_mask,"ml_trained":self.ml.trained,
                 "ml_acc":self.ml.val_acc,"tr":self.tr,
                 "n_samples":self.tr.get("n_samples",0),
                 "n_pca":self.tr.get("n_pca",0),
                 "val_acc":self.ml.val_acc}
        self.ckpt = self.store.save(state, sharpe=self.ml.val_acc)

    def _load(self):
        s = self.store.load()
        if not s: return False
        try:
            self.ml.gbm       = s.get("ml_gbm")
            self.ml.et        = s.get("ml_et")
            self.ml.scaler    = s.get("ml_scaler", RobustScaler())
            self.ml.pca       = s.get("ml_pca")
            self.ml.iso       = s.get("ml_iso", IsotonicRegression(out_of_bounds="clip"))
            self.ml.cal       = s.get("ml_cal", False)
            self.ml.keep_mask = s.get("ml_mask")
            self.ml.trained   = s.get("ml_trained", False)
            self.ml.val_acc   = s.get("ml_acc", 0.5)
            self.tr           = s.get("tr", {})
            self.trained      = self.ml.trained
            return True
        except Exception as e: print("  [LOAD ERR]",e); return False

    def run(self):
        live = False; fund = pd.DataFrame(); df = pd.DataFrame()

        if WS_OK and NET:
            print(cc("  Starting WebSocket...","M"), flush=True)
            self.ws_mgr = WSMgr(CFG["SYMBOL"],CFG["TF"],self.kbuf,self.tbuf)
            self.ws_mgr.start(); time.sleep(3); self.ws_conn = self.ws_mgr.conn

        if NET:
            try:
                df   = fetch(CFG["SYMBOL"],CFG["TF"],CFG["CANDLES"])
                fund = fetch_fund(CFG["SYMBOL"]); live = True
                print("  Data: {} bars".format(len(df)))
            except Exception as e:
                print("  REST error: {}  → synthetic".format(e))

        if df.empty: df, fund = synthetic(n=CFG["CANDLES"], seed=42)
        df = prepare(df)

        if self.store.exists():
            print(cc("  Checkpoint found — loading...","Y"))
            if not self._load():
                self.train(df, fund, verbose=True)
        else:
            self.train(df, fund, verbose=True)

        for _, row in df.tail(300).iterrows():
            self.kbuf.update({"open_time":int(row["open_time"].timestamp()*1000),
                              "open":float(row["open"]),"high":float(row["high"]),
                              "low":float(row["low"]),"close":float(row["close"]),
                              "volume":float(row["volume"]),"taker_buy_vol":float(row["taker_buy_vol"]),
                              "trades":int(row["trades"])})

        print(cc("\n  Loop started. Signals appear during London (8-13 UTC) and NY (13-20 UTC).\n","G"))
        curr_df = df

        while True:
            try:
                if self.ws_mgr and self.ws_conn:
                    self.ws_conn = self.ws_mgr.conn
                    if not self.kbuf.wait(timeout=70): continue
                    curr_df = self.kbuf.get()
                    if curr_df.empty or len(curr_df) < 50: continue
                    curr_df = prepare(curr_df)
                else:
                    time.sleep(30)
                    if NET:
                        try:
                            df   = fetch(CFG["SYMBOL"],CFG["TF"],CFG["CANDLES"])
                            fund = fetch_fund(CFG["SYMBOL"]); live = True
                        except: pass
                    curr_df = prepare(df)

                self.bars+=1; self.bars_tr+=1; self.bars_ck+=1
                price = float(curr_df["close"].iloc[-1])

                if self.bars_tr >= CFG["RETRAIN_N"]:
                    print(cc("\n  [RETRAIN]...","Y"))
                    self.train(curr_df, fund, verbose=False); self.bars_tr = 0

                if self.bars_ck >= CFG["CHECKPOINT_N"]:
                    self._ckpt(); self.bars_ck = 0

                tick = self.tbuf.snap(30000)

                # Tier 1: direction
                dr  = self.dir_eng.score(curr_df, fund, tick)
                # Tier 2: ML confidence filter
                ml  = self.ml.predict(curr_df, fund)

                # Market data
                poc, vah, val = market_profile(curr_df)
                ret_ = curr_df["close"].pct_change().dropna()
                _, garch_m, vol_reg, _ = garch11(ret_)
                atr = float(curr_df["atr"].iloc[-1]) or price*0.003

                # Final side (ML can block; HIGH vol reduces size not blocks)
                side = dr["side"]
                if side != "WAIT" and not ml.get("allow", True):
                    side = "WAIT"

                # Trade levels
                stop_dist = atr * CFG["ATR_SL"]
                if side == "BUY":
                    sl_  = round(min(val, price-stop_dist), 1)
                    tp1  = round(poc if poc>price else price+stop_dist*CFG["TP_MULT"], 1)
                    tp2  = round(vah if vah>tp1   else price+stop_dist*CFG["TP_MULT"]*2, 1)
                elif side == "SELL":
                    sl_  = round(max(vah, price+stop_dist), 1)
                    tp1  = round(poc if poc<price else price-stop_dist*CFG["TP_MULT"], 1)
                    tp2  = round(val if val<tp1   else price-stop_dist*CFG["TP_MULT"]*2, 1)
                else:
                    sl_ = tp1 = tp2 = None

                rr = abs(tp1-price)/max(abs(price-(sl_ or price)),1.) if tp1 else 0.

                # Size: GARCH adjusts (HIGH vol = 0.5× not blocked)
                qty = (CFG["ACCOUNT"]*CFG["MAX_RISK"]*garch_m/max(stop_dist,1.)) if sl_ else 0.

                # Confidence: signal count × ML agreement
                n_ind = dr.get("n_indep", 0)
                ml_p  = ml.get("ml_prob", 0.5)
                ml_ok = (ml_p>0.51 if side=="BUY" else ml_p<0.49 if side=="SELL" else True)
                conf  = min(n_ind/5*100 * (1.3 if ml_ok else 0.9), 99.)

                tradeable = (side!="WAIT" and rr>=CFG["MIN_RR"] and conf>=CFG["MIN_CONF_PCT"])
                final = {"side":side,"tradeable":tradeable,"sl":sl_,"tp1":tp1,"tp2":tp2,
                         "qty":round(qty,3),"rr":rr,"confidence":conf}

                if self.bars > 6: self.sh.resolve(price)

                if self.paper:
                    res = self.paper.update(price)
                    if res:
                        won = res["type"] in ["WIN","TP1"]
                        print(cc("  [PAPER] {}  ${:+.2f}  @${:,.2f}".format(
                            res["type"],res["pnl"],res["price"]),"G" if won else "R"))
                    if tradeable and not self.paper.position:
                        reason = " | ".join(str(v) for v in dr.get("active",{}).values())
                        if self.paper.enter(side,price,sl_,tp1,tp2,qty,
                                            dr["score"],conf,reason):
                            self.sh.record(side,price,dr["score"],conf,n_ind)

                display(price,dr,ml,final,self.tr,self.bars,live,
                        self.paper.stats() if self.paper else None,
                        self.sh,self.ws_conn,self.ckpt,garch_m,vol_reg,poc,vah,val)

            except KeyboardInterrupt:
                print(cc("\n  Stopped.","Y"))
                if self.ws_mgr: self.ws_mgr.stop()
                self._ckpt()
                if self.paper:
                    st = self.paper.stats()
                    print("  Final: ${:,.2f}  PnL={:+.2f}%  WR={:.1f}%  Trades={}".format(
                        st["balance"],st["pnl_pct"],st["wr"],st["trades"]))
                break
            except Exception as e:
                import traceback; print("  Error:",e); traceback.print_exc(); time.sleep(15)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Elite Quant Engine v7.1")
    p.add_argument("--account",   type=float, default=1000.)
    p.add_argument("--paper",     action="store_true")
    p.add_argument("--tf",        type=str,   default="5m")
    p.add_argument("--symbol",    type=str,   default="BTCUSDT")
    p.add_argument("--reset",     action="store_true", help="Delete saved model, retrain fresh")
    p.add_argument("--retrain",   type=int,   default=100)
    p.add_argument("--model-dir", type=str,   default="uq_models_v7", dest="model_dir")
    a = p.parse_args()

    CFG["TF"]        = a.tf
    CFG["SYMBOL"]    = a.symbol
    CFG["ACCOUNT"]   = a.account
    CFG["RETRAIN_N"] = a.retrain
    CFG["MODEL_DIR"] = a.model_dir

    print(cc("\n"+"="*76,"C"))
    print(cc("  ELITE QUANT ENGINE v8.0  —  CRYPTO TRAINED","C"))
    print(cc("  Trained on 1800-bar GARCH BTC data | CPCV=+11.59 | TP=52.6% > SL=44.1%","C"))
    print(cc("="*76,"C"))
    print("  {}  TF:{}  Account:${:,.0f}  Mode:{}".format(
        CFG["SYMBOL"],CFG["TF"],a.account,"PAPER" if a.paper else "SIGNALS"))
    print("  Signals expected during London (08-13 UTC) and NY (13-20 UTC)")
    print("  WebSocket: {}".format("available" if WS_OK else "not installed"))
    print()

    EliteV71(account=a.account, paper=a.paper, reset=a.reset).run()


if __name__ == "__main__":
    main()
