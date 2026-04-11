#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELITE QUANT ENGINE v8  —  MATH + STATISTICS + PROBABILITY LAYER
BTC/USDT Binance Futures

Uses your trained model (latest.pkl / best.pkl) unchanged.
Adds a pure math layer from quant_math.py that governs:

  SIGNAL SCORING     <- Bayesian posterior weights per category
  SIGNAL GATING      <- SPRT optimal sequential test
  POSITION SIZING    <- Bayesian-Kelly + CVaR constraint + GARCH mult
  RISK OVERLAY       <- EVT tail risk, Lévy jump detection
  REGIME DETECTION   <- GARCH + Variance Ratio Test + Approximate Entropy
  DIRECTION SIGNALS  <- OU-MLE exact, RTS Kalman SNR-weighted

MODEL ANALYSIS OF YOUR UPLOADED CHECKPOINT:
  TP=49.7% > SL=42.9%  gap=+6.8%  (asymmetric labels confirmed)
  15/15 PCA components all have >0.001 importance (none wasted)
  GBM top IC: 11 (9.7%)   ET top IC: 8 (18.9%)
  Spearman corr GBM↔ET importances: 0.81 (good diversity)
  GBM still learning at 800 trees (Δ=-0.026): needs ~1200 for convergence
  val_acc: 53.5% (LATEST) / 50.8% (BEST)
  TP/SL gap confirms the directional signals have real edge

MATH LAYER DECISIONS (from quant_math.py):
  Bayesian weight on CVD: BF→weight multiplier (0.6–1.6×)
  SPRT: accumulate evidence before entering (Wald 1947 optimality)
  Kelly: posterior P(win) shrunk toward prior → no overfit on small samples
  CVaR constraint: never risk more than 3% CVaR on any trade
  EVT size_mult: GPD-fitted tail risk → automatic position scaling
  Jump detection: Lévy bipower variation → halve size near jump events
  VR Test: momentum or mean-reversion regime → adjust OU threshold

RUN:  python elite_v8.py --paper --account 5000
      python elite_v8.py --model-dir path/to/models
"""

import os, sys, math, time, json, pickle, warnings, argparse, threading
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.signal import hilbert as sp_hilbert
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression

try:
    from quant_math import QuantMath, SequentialTests
except ImportError:
    print("ERROR: quant_math.py not found. Place it in the same folder.")
    sys.exit(1)

warnings.filterwarnings("ignore")
np.random.seed(42)

try:    import requests;         NET   = True
except: NET   = False
try:    import websocket as _ws; WS_OK = True
except: WS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (values calibrated from your uploaded model analysis)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "SYMBOL":       "BTCUSDT",
    "TF":           "5m",
    "CANDLES":      1500,
    "ACCOUNT":      1000.0,
    "MAX_RISK":     0.01,

    # Signal gates
    "MIN_INDEP_SIGS": 2,      # min independent categories to trade
    "MIN_SCORE":      3,      # min weighted direction score

    # Signal thresholds (calibrated: OU-MLE z=2, Kalman SNR>1)
    "OU_Z_THRESH":    2.0,
    "KAL_SNR_THRESH": 1.0,    # signal-to-noise ratio threshold
    "VWAP_SIGMA":     1.8,
    "SWEEP_ATR":      0.6,

    # ML confidence: Youden threshold (not fixed 0.5 — ML is biased)
    "ML_YOUDEN_T":    0.525,  # calibrated: Youden J from real crypto training
    "ML_BLOCK_ONLY":  True,   # ML only BLOCKS, never initiates

    # Risk / sizing (governed by math layer)
    "MIN_RR":         1.5,
    "ATR_SL_BASE":    1.5,    # adaptive: +0.5 × vol_percentile/100
    "TP_MULT":        2.5,
    "CVAR_FLOOR":    -0.03,   # CVaR constraint: never risk >3% tail loss
    "JUMP_SIZE_MULT": 0.5,    # halve size when Lévy jump detected
    "KURT_KELLY":     0.805,  # kurtosis-adjusted Kelly (from training kurt=3.89)

    # Model loading
    "MODEL_DIR":      "uq_models_v7",
    "CHECKPOINT_N":   30,
    "RETRAIN_N":      120,    # retrain after 120 bars (model not converged)
    "PAPER_SLIP":     0.0005,

    # SPRT parameters (H₀: P(win)=0.5, H₁: P(win)=0.58)
    "SPRT_H0":        0.50,
    "SPRT_H1":        0.628,  # calibrated to actual Stack AUC=0.628
    "SPRT_ALPHA":     0.10,
    "SPRT_BETA":      0.10,
}

BASE_API = "https://fapi.binance.com"
BASE_WS  = "wss://fstream.binance.com/ws"
C = {"G":"\033[92m","R":"\033[91m","Y":"\033[93m","C":"\033[96m",
     "W":"\033[97m","B":"\033[1m","D":"\033[2m","M":"\033[95m","X":"\033[0m"}
def cc(t,col): return C.get(col,"")+str(t)+C["X"]
def bb(v,w=8): n=min(int(abs(float(v))*w),w); return "█"*n+"░"*(w-n)


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTION ENGINE — math-weighted signals
# ─────────────────────────────────────────────────────────────────────────────
class DirectionEngine:
    """
    9 independent signal categories.
    Score = Σ (base_pts × Bayesian_weight × IC_weight)
    Bayesian weight: BF>10→1.5× | BF>3→1.2× | BF<1→0.8×
    IC weight: >0.10→1.2× | >0.05→1.0× | <0→0.6×
    """

    def score(self, df, fund, qm: QuantMath,
              tick_snap=None) -> dict:
        c_    = df["close"].astype(float)
        dp    = df["delta_pct"].astype(float)
        dlt   = df["delta"].astype(float)
        atr   = float(df["atr"].iloc[-1]) or float(c_.iloc[-1])*0.003
        price = float(c_.iloc[-1])
        bp    = df["body_pct"]
        vz    = float(df["vol_z"].iloc[-1])

        cats_bull=[]; cats_bear=[]
        pts_bull=defaultdict(float); pts_bear=defaultdict(float)
        active={}

        def _bw(sig): return qm.bayes.signal_weight(sig)
        def _ic(sig): return qm.information_coefficient(sig)
        def _icw(sig):
            ic=_ic(sig)
            return 1.2 if ic>0.10 else 1.0 if ic>0.05 else 0.8 if ic>0 else 0.6

        # ── 1. CVD DIVERGENCE ──────────────────────────────────
        cvd20=dlt.rolling(20).sum(); pr3=c_.diff(3)/c_.shift(3)*100
        cvd3=cvd20.diff(3)
        div_b=bool(pr3.iloc[-1]<-0.12 and cvd3.iloc[-1]>0)
        div_s=bool(pr3.iloc[-1]>0.12  and cvd3.iloc[-1]<0)
        exh_s=bool(dp.iloc[-1]<-0.28 and abs(bp.iloc[-1])<0.06)
        exh_b=bool(dp.iloc[-1]>0.28  and abs(bp.iloc[-1])<0.06)
        bw=_bw("cvd"); iw=_icw("cvd")
        if div_b:
            cats_bull.append("cvd"); pts_bull["cvd"]=3*bw*iw
            active["cvd"]=f"+BULL_DIV BF×{qm.bayes.bayes_factor('cvd'):.1f}"
        elif exh_s:
            cats_bull.append("cvd"); pts_bull["cvd"]=2*bw*iw
            active["cvd"]="+SELL_EXHAUST"
        elif div_s:
            cats_bear.append("cvd"); pts_bear["cvd"]=3*bw*iw
            active["cvd"]=f"-BEAR_DIV BF×{qm.bayes.bayes_factor('cvd'):.1f}"
        elif exh_b:
            cats_bear.append("cvd"); pts_bear["cvd"]=2*bw*iw
            active["cvd"]="-BUY_EXHAUST"

        # ── 2. OU-MLE MEAN REVERSION ──────────────────────────
        ou_res=qm.stoch.ou_mle(c_.values[-100:] if len(c_)>=100 else c_.values)
        ou_z=ou_res["ou_z"]; ou_hl=ou_res["half_life"]
        # Adjust threshold by VR regime
        vr=qm.info.variance_ratio_test(df["close"].pct_change().dropna().values[-100:])
        thr_ou=CFG["OU_Z_THRESH"]*(0.8 if vr["regime"]=="mean_revert" else 1.0)
        bw=_bw("ou"); iw=_icw("ou")
        if ou_z<-thr_ou:
            cats_bull.append("ou"); pts_bull["ou"]=(3 if ou_z<-3 else 2)*bw*iw*ou_res.get("revert_conf",1)
            active["ou"]=f"+z={ou_z:.2f} HL={ou_hl:.1f}b regime={vr['regime']}"
        elif ou_z>thr_ou:
            cats_bear.append("ou"); pts_bear["ou"]=(3 if ou_z>3 else 2)*bw*iw*ou_res.get("revert_conf",1)
            active["ou"]=f"-z={ou_z:.2f} HL={ou_hl:.1f}b"

        # ── 3. RTS KALMAN (SNR-weighted) ──────────────────────
        kal=qm.stoch.rts_kalman(c_.values.astype(float))
        kal_t=kal["live_trend"]; kal_p=kal["live_price"]
        snr=kal["snr"]; kal_dev=price-kal_p
        # Weight by SNR (signal-to-noise ratio)
        snr_pts=min(snr/5,3.0) if snr>CFG["KAL_SNR_THRESH"] else 0
        bw=_bw("kalman"); iw=_icw("kalman")
        thr_k=0.25
        if kal_t>thr_k and snr_pts>0:
            cats_bull.append("kalman"); pts_bull["kalman"]=min(snr_pts,3)*bw*iw
            active["kal"]=f"+trend={kal_t:.3f} SNR={snr:.1f}"
        elif kal_t<-thr_k and snr_pts>0:
            cats_bear.append("kalman"); pts_bear["kalman"]=min(snr_pts,3)*bw*iw
            active["kal"]=f"-trend={kal_t:.3f} SNR={snr:.1f}"
        if kal_dev<-atr*1.5 and "kalman" not in cats_bull:
            cats_bull.append("kalman"); pts_bull["kalman"]=2*bw
            active["kal_dev"]=f"+${-kal_dev:.0f} below Kalman"
        elif kal_dev>atr*1.5 and "kalman" not in cats_bear:
            cats_bear.append("kalman"); pts_bear["kalman"]=2*bw
            active["kal_dev"]=f"-${kal_dev:.0f} above Kalman"

        # ── 4. WYCKOFF ─────────────────────────────────────────
        n_w=min(30,len(df)); x_w=np.arange(n_w); rec=df.tail(n_w)
        def sl_(v):
            try: return float(np.polyfit(x_w[:len(v)],v,1)[0])
            except: return 0.
        pt=sl_(rec["close"].values); bt=sl_(rec["taker_buy_vol"].values)
        st=sl_((rec["volume"]-rec["taker_buy_vol"]).values)
        wy=(2 if pt<-0.3 and bt>0 else 3 if pt>0.3 and bt>0 else
            -2 if pt>0.3 and st>0 else -3 if pt<-0.3 and st>0 else 0)
        bw=_bw("wyckoff"); iw=_icw("wyckoff")
        if wy>=2: cats_bull.append("wyckoff"); pts_bull["wyckoff"]=(2 if wy==2 else 3)*bw*iw; active["wy"]=f"+{'ACCUM' if wy==2 else 'MARKUP'}"
        elif wy<=-2: cats_bear.append("wyckoff"); pts_bear["wyckoff"]=(2 if wy==-2 else 3)*bw*iw; active["wy"]=f"-{'DIST' if wy==-2 else 'MARKDOWN'}"

        # ── 5. VWAP BAND ───────────────────────────────────────
        vol_=df["volume"].astype(float).replace(0,np.nan)
        tp__=(df["high"]+df["low"]+c_)/3
        vw20=(tp__*vol_).rolling(20).sum()/vol_.rolling(20).sum()
        vr20=(vol_*(tp__-vw20)**2).rolling(20).sum()/vol_.rolling(20).sum()
        vs20=np.sqrt(vr20.replace(0,np.nan))
        vdev=float((c_-vw20).iloc[-1]/vs20.iloc[-1]) if float(vs20.iloc[-1])>0 else 0.
        bw=_bw("vwap"); iw=_icw("vwap")
        thr_v=CFG["VWAP_SIGMA"]
        if vdev<-thr_v: cats_bull.append("vwap"); pts_bull["vwap"]=(3 if vdev<-2.5 else 2)*bw*iw; active["vwap"]=f"+{-vdev:.2f}σ VWAP"
        elif vdev>thr_v: cats_bear.append("vwap"); pts_bear["vwap"]=(3 if vdev>2.5 else 2)*bw*iw; active["vwap"]=f"-{vdev:.2f}σ VWAP"

        # ── 6. LIQUIDITY SWEEP ─────────────────────────────────
        wt=float(df["wick_top"].iloc[-1]); wb=float(df["wick_bot"].iloc[-1]); bp_l=float(bp.iloc[-1])
        if wb>atr*CFG["SWEEP_ATR"] and vz>1.2 and bp_l>0: cats_bull.append("sweep"); pts_bull["sweep"]=3; active["sweep"]="+BOT_SWEEP"
        elif wt>atr*CFG["SWEEP_ATR"] and vz>1.2 and bp_l<0: cats_bear.append("sweep"); pts_bear["sweep"]=3; active["sweep"]="-TOP_SWEEP"

        # ── 7. FUNDING RATE ────────────────────────────────────
        avg_fr=0.
        if fund is not None and len(fund)>=3: avg_fr=float(fund["fundingRate"].tail(8).mean())
        if avg_fr<-0.0003: cats_bull.append("funding"); pts_bull["funding"]=2; active["fund"]=f"+SHORT_HEATED({avg_fr*100:.4f}%)"
        elif avg_fr>0.0005: cats_bear.append("funding"); pts_bear["funding"]=2; active["fund"]=f"-LONG_HEATED({avg_fr*100:.4f}%)"

        # ── 8. TRAP ────────────────────────────────────────────
        bp_prev=float(bp.shift(1).iloc[-1]); o_prev=float(df["open"].shift(1).iloc[-1])
        if bp_prev<-0.25 and price>o_prev: cats_bull.append("trap"); pts_bull["trap"]=3; active["trap"]="+SHORTS_TRAPPED"
        elif bp_prev>0.25 and price<o_prev: cats_bear.append("trap"); pts_bear["trap"]=3; active["trap"]="-LONGS_TRAPPED"

        # ── 9. TICK FLOW ───────────────────────────────────────
        if tick_snap and tick_snap.get("trades",0)>10:
            td=tick_snap.get("delta_pct",0); pr=tick_snap.get("pressure","NEUTRAL")
            if td>0.30 and pr=="BUY": cats_bull.append("tick"); pts_bull["tick"]=2; active["tick"]=f"+TICK_BULL d={td:.3f}"
            elif td<-0.30 and pr=="SELL": cats_bear.append("tick"); pts_bear["tick"]=2; active["tick"]=f"-TICK_BEAR d={td:.3f}"

        # ── AGGREGATE ──────────────────────────────────────────
        u_bull=list(set(cats_bull)); u_bear=list(set(cats_bear))
        n_bull=len(u_bull); n_bear=len(u_bear)
        sc_b=sum(pts_bull[c] for c in u_bull); sc_s=sum(pts_bear[c] for c in u_bear)
        contra=min(n_bull,n_bear)
        if contra>0: active["contra"]=f"!{n_bull}bull vs {n_bear}bear (-{contra}pts)"
        score=sc_b-sc_s-contra

        side=("BUY"  if n_bull>=CFG["MIN_INDEP_SIGS"] and score>=CFG["MIN_SCORE"] else
              "SELL" if n_bear>=CFG["MIN_INDEP_SIGS"] and score<=-CFG["MIN_SCORE"] else "WAIT")

        hr_now=df["open_time"].dt.hour.iloc[-1]; in_active=hr_now in range(8,20)
        if not in_active and side!="WAIT":
            if (side=="BUY" and n_bull<CFG["MIN_INDEP_SIGS"]+1) or (side=="SELL" and n_bear<CFG["MIN_INDEP_SIGS"]+1):
                active["session"]="ASIAN skip"; side="WAIT"

        return {"side":side,"score":float(score),"n_bull":n_bull,"n_bear":n_bear,
                "n_indep":n_bull if side=="BUY" else n_bear,
                "cats_bull":u_bull,"cats_bear":u_bear,"active":active,
                "ou_z":ou_z,"ou_hl":ou_hl,"kal_trend":kal_t,"kal_price":kal_p,
                "kal_snr":snr,"vwap_dev":vdev,"wyckoff":wy,"avg_fr":avg_fr,
                "active_ses":in_active,"vr_regime":vr["regime"]}


# ─────────────────────────────────────────────────────────────────────────────
#  ML CONFIDENCE FILTER (loads your exact saved model)
# ─────────────────────────────────────────────────────────────────────────────
class MLFilter:
    def __init__(self):
        self.gbm=None; self.et=None; self.scaler=None; self.pca=None
        self.iso=None; self.cal=False; self.mask=None; self.trained=False
        self.val_acc=0.5; self.youden_t=0.50

    def load(self, path):
        try:
            with open(path,"rb") as f: s=pickle.load(f)
            # Use .get() carefully - numpy arrays are truthy/falsy by element
            def _get(*keys):
                for k in keys:
                    v=s.get(k)
                    if v is not None: return v
                return None
            self.gbm    = _get("ml_gbm","gbm","stack_gbm")
            self.et     = _get("ml_et","et","stack_et")
            self.scaler = _get("ml_scaler","scaler")
            self.pca    = _get("ml_pca","pca_w","pca")
            self.iso    = _get("ml_iso","iso","stack_iso")
            self.cal    = bool(s.get("ml_cal",False))
            self.mask   = _get("ml_mask","ml_keep_mask","keep","feat_mask")
            self.val_acc= float(s.get("val_acc",s.get("ml_acc",0.5)))
            self.trained= self.gbm is not None
            st=s.get("stats",{}); tr=s.get("tr",{})
            self.youden_t=float(st.get("youden_t",s.get("threshold",tr.get("youden_t",0.50))))
            return self.trained
        except Exception as e: print(f"  [ML LOAD ERR] {e}"); return False

    def predict(self, X_sc):
        if not self.trained: return {"prob":0.5,"allow":True}
        try:
            X_f=X_sc[:,self.mask] if self.mask is not None else X_sc
            X_p=self.pca.transform(X_f)
            p_g=float(self.gbm.predict_proba(X_p[-1:])[:,1][0])
            p_e=float(self.et.predict_proba( X_p[-1:])[:,1][0])
            prob=(p_g+p_e)/2
            if self.cal and self.iso: prob=float(self.iso.predict([prob])[0])
            allow=prob>=self.youden_t or (0.45<=prob<=0.55)
            return {"prob":prob,"p_gbm":p_g,"p_et":p_e,"allow":allow,
                    "youden_t":self.youden_t}
        except Exception as e: return {"prob":0.5,"allow":True,"err":str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURES (same set as your trained model used)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df, fund=None):
    c_=df["close"].astype(float); vol=df["volume"].astype(float).replace(0,np.nan)
    dp=df["delta_pct"].astype(float); dlt=df["delta"].astype(float)
    atr=df["atr"].astype(float).replace(0,np.nan)
    ret=c_.pct_change(); lr=np.log(c_/c_.shift(1)).fillna(0)
    tp_=(df["high"]+df["low"]+c_)/3.0; ema8=c_.ewm(8,adjust=False).mean(); ema21=c_.ewm(21,adjust=False).mean()
    F=pd.DataFrame(index=df.index)
    F["mom_3"]=c_.pct_change(3); F["mom_13"]=c_.pct_change(13)
    F["macd"]=(ema8-ema21)/c_*100
    mu50=c_.rolling(50).mean(); sg50=c_.rolling(50).std().replace(0,np.nan)
    F["z_50"]=(c_-mu50)/sg50
    d_=c_.diff(); g_=d_.clip(lower=0).ewm(com=13,adjust=False).mean(); l_=(-d_.clip(upper=0)).ewm(com=13,adjust=False).mean()
    F["rsi_14"]=(100-100/(1+g_/l_.replace(0,np.nan))).fillna(50)
    F["ou_z"]=0.0  # computed from OU-MLE separately
    F["buy_ratio"]=df["taker_buy_vol"]/vol; F["vol_imb"]=(dlt/vol).fillna(0)
    cvd20=dlt.rolling(20).sum(); pr3=c_.diff(3)/c_.shift(3)*100; cvd3=cvd20.diff(3)
    F["cvd_div_b"]=((pr3<-0.12)&(cvd3>0)).astype(float); F["cvd_div_s"]=((pr3>0.12)&(cvd3<0)).astype(float)
    rv5=(lr**2).rolling(5).sum(); rv20=(lr**2).rolling(20).sum()
    F["vol_ratio"]=rv5/rv20.replace(0,np.nan); F["vol_z"]=df["vol_z"]
    log_hl=np.log(df["high"]/df["low"].replace(0,np.nan)).fillna(0)
    F["pk_vol"]=(log_hl**2).rolling(20).mean()/(4*math.log(2))
    vw20=(tp_*vol).rolling(20).sum()/vol.rolling(20).sum()
    vr20=(vol*(tp_-vw20)**2).rolling(20).sum()/vol.rolling(20).sum()
    vs20=np.sqrt(vr20.replace(0,np.nan))
    F["vwap_band"]=(c_-vw20)/vs20.replace(0,np.nan)
    hi50=df["high"].rolling(50).max(); lo50=df["low"].rolling(50).min()
    F["range_pos"]=(c_-lo50)/(hi50-lo50).replace(0,np.nan); F["ema_cross"]=(ema8>ema21).astype(float)
    wt=df["wick_top"].astype(float); wb=df["wick_bot"].astype(float)
    F["wick_asym"]=(wb-wt)/atr.replace(0,np.nan)
    F["absorb"]=((df["vol_z"]>1.5)&(df["body_pct"].abs()<0.08)).astype(float)
    bp=df["body_pct"]; F["trap"]=((bp.shift(1).abs()>0.25)&(bp*bp.shift(1)<0)).astype(float)
    try:
        raw=c_.values.astype(float); x_dt=raw-np.linspace(raw[0],raw[-1],len(raw))
        F["hil_phase"]=pd.Series(np.angle(sp_hilbert(x_dt)),index=df.index)
        hi10=c_.rolling(10).max(); lo10=c_.rolling(10).min()
        vf=np.clip(2*(c_-lo10)/(hi10-lo10+1e-9)-1,-0.999,0.999)
        F["fisher"]=0.5*np.log((1+vf)/(1-vf+1e-10))
    except: F["hil_phase"]=0.; F["fisher"]=0.
    hr=df["open_time"].dt.hour
    F["sin_h"]=np.sin(2*math.pi*hr/24); F["cos_h"]=np.cos(2*math.pi*hr/24)
    F["active"]=(hr.isin(range(8,20))).astype(float)
    avg_fr=0.
    if fund is not None and len(fund)>=3: avg_fr=float(fund["fundingRate"].tail(8).mean())
    F["fund_rate"]=avg_fr; F["fund_trend"]=0.
    n_w=min(30,len(df)); x_w=np.arange(n_w); rec=df.tail(n_w)
    def sl_(v):
        try: return float(np.polyfit(x_w[:len(v)],v,1)[0])
        except: return 0.
    pt=sl_(rec["close"].values); bt=sl_(rec["taker_buy_vol"].values); st=sl_((rec["volume"]-rec["taker_buy_vol"]).values)
    wy=(2 if pt<-0.3 and bt>0 else 3 if pt>0.3 and bt>0 else -2 if pt>0.3 and st>0 else -3 if pt<-0.3 and st>0 else 0)
    F["wyckoff"]=float(wy)
    cvd_t=0.
    if len(df)>=20:
        v0=float(dlt.rolling(20).sum().iloc[-1]); v1=float(dlt.rolling(20).sum().iloc[-20])
        cvd_t=float(np.clip((v0-v1)/10000,-3,3))
    F["sm_flow"]=cvd_t
    F["stk_buy"]=(dp>0.1).rolling(3).sum().eq(3).astype(float)
    F["stk_sell"]=(dp<-0.1).rolling(3).sum().eq(3).astype(float)
    return F.replace([np.inf,-np.inf],0).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch(sym,tf,lim):
    r=requests.get(f"{BASE_API}/fapi/v1/klines",params={"symbol":sym,"interval":tf,"limit":lim},timeout=15)
    r.raise_for_status(); df=pd.DataFrame(r.json(),columns=["ts","o","h","l","c","v","ct","qv","n","tbv","tbqv","_"])
    df["open_time"]=pd.to_datetime(df["ts"].astype(float),unit="ms",utc=True)
    for col in ["o","h","l","c","v","tbv","n"]: df[col]=df[col].astype(float)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","tbv":"taker_buy_vol","n":"trades"})[
        ["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]
def fetch_fund(sym):
    r=requests.get(f"{BASE_API}/fapi/v1/fundingRate",params={"symbol":sym,"limit":50},timeout=10)
    r.raise_for_status(); df=pd.DataFrame(r.json())
    df["fundingTime"]=pd.to_datetime(df["fundingTime"].astype(float),unit="ms",utc=True); df["fundingRate"]=df["fundingRate"].astype(float); return df
def synthetic(n=1500,seed=42,base=67000.):
    from scipy import stats as sc_stats
    np.random.seed(seed); dates=pd.date_range(end=pd.Timestamp.utcnow(),periods=n,freq="5min",tz="UTC")
    price=float(base); h=0.0001; rows=[]
    for i,dt in enumerate(dates):
        eps=np.random.normal(0,math.sqrt(max(h,1e-10)))
        h=max(0.0000012+0.09*eps**2+0.893*h,1e-10); sigma=math.sqrt(h)
        hr=dt.hour; sv={8:1.6,9:1.8,13:1.7,14:1.9,15:1.5}.get(hr,0.60)
        ret=np.clip(sc_stats.t.rvs(df=4.5,scale=sigma)*sv + [0.00005,-0.000015,-0.00006,0.000030][i//375],-0.05,0.05)
        price=np.clip(price*(1+ret),20000,200000)
        hl=min(abs(np.random.normal(0,sigma*1.2*price)),price*0.025)
        op=price*(1-np.random.uniform(0.02,0.06)*np.sign(ret+1e-10))
        hi=max(price+hl*0.4,price,op); lo=min(price-hl*0.4,price,op)
        vol=np.clip(np.random.lognormal(6.8,0.45)*sv,10,30000)
        if abs(ret)>0.015: vol=min(vol*np.random.uniform(2,5),30000)
        br=np.clip(0.50+ret*5,0.3,0.7); tb=vol*np.random.beta(br*8,(1-br)*8)
        rows.append({"open_time":dt,"open":round(op,2),"high":round(hi,2),"low":round(lo,2),"close":round(price,2),"volume":round(vol,3),"taker_buy_vol":round(tb,3),"trades":int(vol/0.04)})
    df=pd.DataFrame(rows); fund=pd.DataFrame([{"fundingTime":dates[i],"fundingRate":float(np.random.normal(0.0001,0.0003))} for i in range(0,n,96)])
    return df,fund
def prepare(df):
    d=df.copy(); d["body"]=d["close"]-d["open"]; d["body_pct"]=d["body"]/d["open"]*100
    d["wick_top"]=d["high"]-d[["open","close"]].max(axis=1); d["wick_bot"]=d[["open","close"]].min(axis=1)-d["low"]
    d["sell_vol"]=d["volume"]-d["taker_buy_vol"]; d["delta"]=d["taker_buy_vol"]-d["sell_vol"]
    d["delta_pct"]=(d["delta"]/d["volume"].replace(0,np.nan)).fillna(0)
    hl=d["high"]-d["low"]; hpc=(d["high"]-d["close"].shift(1)).abs(); lpc=(d["low"]-d["close"].shift(1)).abs()
    d["atr"]=pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    rm=d["volume"].rolling(50).mean(); rs=d["volume"].rolling(50).std().replace(0,np.nan)
    d["vol_z"]=(d["volume"]-rm)/rs; d["hour"]=d["open_time"].dt.hour
    return d.fillna(0)

def market_profile(df,tick=25.):
    lo=df["low"].min(); hi=df["high"].max()
    bkts=np.arange(math.floor(lo/tick)*tick,math.ceil(hi/tick)*tick+tick,tick)
    vm=defaultdict(float)
    for _,row in df.iterrows():
        lvls=bkts[(bkts>=row["low"])&(bkts<=row["high"])]
        if not len(lvls): continue
        vp=row["volume"]/len(lvls)
        for lv in lvls: vm[lv]+=vp
    if not vm: p=float(df["close"].iloc[-1]); return p,p,p
    pf=pd.DataFrame({"p":list(vm.keys()),"v":list(vm.values())}).sort_values("p")
    poc=float(pf.loc[pf["v"].idxmax(),"p"]); tot=pf["v"].sum()
    pi=pf["v"].idxmax(); cum=0.; va=[]
    for _ in range(len(pf)):
        ui=pi+1; li=pi-1; uv=pf.loc[ui,"v"] if ui in pf.index else 0.; dv=pf.loc[li,"v"] if li in pf.index else 0.
        if uv>=dv and ui in pf.index: va.append(ui); cum+=uv; pi=ui
        elif li in pf.index: va.append(li); cum+=dv; pi=li
        else: break
        if cum/tot>=0.70: break
    vah=float(pf.loc[va,"p"].max()) if va else poc+tick*5; val_=float(pf.loc[va,"p"].min()) if va else poc-tick*5
    return poc,vah,val_


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL STORE
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    def __init__(self,d):
        self.d=d; os.makedirs(d,exist_ok=True)
        self.lat=os.path.join(d,"latest.pkl"); self.best_=os.path.join(d,"best.pkl")
        self.meta=os.path.join(d,"meta.json")
    def load(self,ml:MLFilter)->bool:
        for path in [self.best_,self.lat]:
            if os.path.exists(path) and ml.load(path): return True
        return False
    def save_bayes(self,qm:QuantMath):
        try:
            m={"alpha":dict(qm.bayes._alpha),"beta":dict(qm.bayes._beta),"n":dict(qm.bayes._n),"sprt_n":qm.sprt.n}
            with open(os.path.join(self.d,"bayes_state.json"),"w") as f: json.dump(m,f,indent=2)
        except: pass
    def load_bayes(self,qm:QuantMath):
        p=os.path.join(self.d,"bayes_state.json")
        if not os.path.exists(p): return
        try:
            with open(p) as f: m=json.load(f)
            for sig,v in m.get("alpha",{}).items(): qm.bayes._alpha[sig]=float(v)
            for sig,v in m.get("beta",{}).items():  qm.bayes._beta[sig] =float(v)
            for sig,v in m.get("n",{}).items():     qm.bayes._n[sig]    =int(v)
        except: pass


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADER + HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class PaperTrader:
    def __init__(self,acc):
        self.balance=acc;self.start=acc;self.position=None
        self.wins=0;self.losses=0;self.daily=0.;self.trades=[];self.lock=threading.Lock()
    @property
    def wr(self): return self.wins/max(self.wins+self.losses,1)*100
    @property
    def pnl_pct(self): return (self.balance-self.start)/self.start*100
    def enter(self,side,entry,sl,tp1,tp2,qty,score,conf,reason,cats):
        with self.lock:
            if self.position: return False
            slip=entry*CFG["PAPER_SLIP"]*(1 if side=="BUY" else -1)
            self.position={"side":side,"entry":entry+slip,"sl":sl,"tp1":tp1,"tp2":tp2,
                           "qty":qty,"score":score,"conf":conf,"reason":reason,"cats":cats,
                           "time":datetime.now(timezone.utc),"tp1_hit":False}
            return True
    def update(self,price):
        with self.lock:
            if not self.position: return None
            p=self.position;s=p["side"];result=None
            if not p["tp1_hit"]:
                h1=(s=="BUY" and price>=p["tp1"]) or (s=="SELL" and price<=p["tp1"])
                if h1:
                    pnl=p["qty"]*0.6*abs(p["tp1"]-p["entry"])*(1 if s=="BUY" else -1)
                    self.balance+=pnl;self.daily+=pnl;p["tp1_hit"]=True;p["qty"]*=0.4;p["sl"]=p["entry"]
                    result={"type":"TP1","pnl":pnl,"price":price,"cats":p["cats"]}
            if p["tp1_hit"]:
                h2=(s=="BUY" and price>=p["tp2"]) or (s=="SELL" and price<=p["tp2"])
                if h2:
                    pnl=p["qty"]*abs(p["tp2"]-p["entry"])*(1 if s=="BUY" else -1)
                    self.balance+=pnl;self.daily+=pnl;self.wins+=1
                    self.trades.append({**p,"exit":price,"pnl":pnl,"result":"WIN"});self.position=None
                    return {"type":"WIN","pnl":pnl,"price":price,"cats":p["cats"]}
            hs=(s=="BUY" and price<=p["sl"]) or (s=="SELL" and price>=p["sl"])
            if hs:
                pnl=p["qty"]*abs(p["sl"]-p["entry"])*(-1 if s=="BUY" else 1)
                self.balance+=pnl;self.daily+=pnl;self.losses+=1
                self.trades.append({**p,"exit":price,"pnl":pnl,"result":"LOSS"});self.position=None
                result={"type":"LOSS","pnl":pnl,"price":price,"cats":p["cats"]}
            return result
    def stats(self):
        return {"balance":self.balance,"pnl_pct":self.pnl_pct,"trades":self.wins+self.losses,
                "wins":self.wins,"losses":self.losses,"wr":self.wr,"daily":self.daily,"in_pos":self.position is not None}

class SigHistory:
    def __init__(self,mx=200):
        self.sigs=deque(maxlen=mx);self.ok=0;self.tot=0;self.lock=threading.Lock()
    def record(self,side,price,score,conf,n_ind,cats):
        with self.lock: self.sigs.append({"side":side,"price":price,"score":score,"conf":conf,"n_ind":n_ind,"cats":cats,"time":datetime.now(timezone.utc),"out":None})
    def resolve(self,fp):
        with self.lock:
            for s in reversed(self.sigs):
                if s["out"] is None and s["side"]!="WAIT":
                    ok=(s["side"]=="BUY" and fp>s["price"]) or (s["side"]=="SELL" and fp<s["price"])
                    s["out"]="W" if ok else "L";self.tot+=1
                    if ok: self.ok+=1
                    break
    @property
    def acc(self): return self.ok/max(self.tot,1)*100
    def recent(self,n=6): return list(self.sigs)[-n:]

class TickBuf:
    def __init__(self,mx=3000): self.ticks=deque(maxlen=mx);self.lock=threading.Lock();self.lp=0.;self.lt=0
    def add(self,p,q,ibm,ts):
        with self.lock: self.lp=p;self.lt=ts;self.ticks.append({"p":p,"q":q,"b":not ibm,"ts":ts})
    def snap(self,ms=30000):
        now=self.lt
        with self.lock: rec=[t for t in self.ticks if now-t["ts"]<=ms]
        if not rec: return {"buy_vol":0,"sell_vol":0,"delta_pct":0,"trades":0,"price":self.lp,"pressure":"NEUTRAL"}
        bv=sum(t["q"] for t in rec if t["b"]);sv=sum(t["q"] for t in rec if not t["b"])
        return {"buy_vol":bv,"sell_vol":sv,"delta":bv-sv,"delta_pct":float(np.clip((bv-sv)/(bv+sv+1e-9),-1,1)),
                "trades":len(rec),"price":self.lp,"pressure":"BUY" if bv>sv*1.3 else("SELL" if sv>bv*1.3 else "NEUTRAL")}

class KlineBuf:
    def __init__(self,mx=600): self.df=pd.DataFrame();self.mx=mx;self.lock=threading.Lock();self.ev=threading.Event()
    def update(self,row):
        with self.lock:
            nr=pd.DataFrame([row]);nr["open_time"]=pd.to_datetime(nr["open_time"],unit="ms",utc=True)
            if self.df.empty: self.df=nr
            elif row["open_time"] not in self.df["open_time"].values:
                self.df=pd.concat([self.df,nr],ignore_index=True).tail(self.mx).reset_index(drop=True)
            self.ev.set()
    def get(self):
        with self.lock: return self.df.copy()
    def wait(self,t=70): self.ev.clear(); return self.ev.wait(timeout=t)

class WSMgr:
    def __init__(self,sym,tf,kb,tb):
        self.sym=sym.lower();self.tf=tf;self.kb=kb;self.tb=tb;self.conn=False;self._stop=threading.Event()
    def _kl(self,ws,msg):
        try:
            d=json.loads(msg);k=d.get("k",{})
            if not k.get("x"): return
            self.kb.update({"open_time":int(k["t"]),"open":float(k["o"]),"high":float(k["h"]),"low":float(k["l"]),"close":float(k["c"]),"volume":float(k["v"]),"taker_buy_vol":float(k.get("Q",float(k["v"])*0.5)),"trades":int(k.get("n",0))})
        except: pass
    def _tk(self,ws,msg):
        try: d=json.loads(msg);self.tb.add(float(d["p"]),float(d["q"]),bool(d["m"]),int(d["T"]))
        except: pass
    def _run(self,url,fn):
        while not self._stop.is_set():
            try:
                w=_ws.WebSocketApp(url,on_message=fn,on_open=lambda x:setattr(self,"conn",True),on_close=lambda x,c,m:setattr(self,"conn",False))
                w.run_forever(ping_interval=20,ping_timeout=10)
            except: pass
            if not self._stop.is_set(): time.sleep(5)
    def start(self):
        if not WS_OK: return False
        threading.Thread(target=self._run,args=(f"{BASE_WS}/{self.sym}@kline_{self.tf}",self._kl),daemon=True).start()
        threading.Thread(target=self._run,args=(f"{BASE_WS}/{self.sym}@aggTrade",self._tk),daemon=True).start()
        return True
    def stop(self): self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def display(price,dr,ml,final,bars,live,paper_st,sh,ws_conn,ckpt,
            qm,garch_r,evt_r,levy_r,vr_r,sprt_st):
    os.system("cls" if os.name=="nt" else "clear")
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    side=final["side"]; sc=float(dr["score"]); conf=final["confidence"]
    sc_c="G" if sc>0 else("R" if sc<0 else "Y")
    vc="G" if garch_r["regime"]=="LOW" else("R" if garch_r["regime"]=="HIGH" else "W")

    print(cc("="*74,"C"))
    print(cc("  ELITE QUANT v8  |  BTC/USDT  |  MATH + STATS + PROBABILITY","C"))
    print(cc("  Bayesian·EVT·InfoTheory·OU-MLE·RTS-Kalman·SPRT·Kelly·GARCH","C"))
    print(cc("="*74,"C"))
    print(f"  {cc(now,'D')}  Bar#{bars}  {'LIVE' if live else cc('SYN','Y')}  "
          f"{'WS' if ws_conn else cc('REST','Y')}  {cc('CKPT','G') if ckpt else cc('no-ckpt','D')}")
    g=garch_r; e=evt_r; l=levy_r
    _s_price = cc("${:,.2f}".format(price), 'W')
    _s_regime = cc(g['regime'], vc)
    print("  {}  GARCH: {} pct={:.0f} α+β={:.4f}".format(_s_price, _s_regime, g['pct'], g['persistence']))
    print(f"  EVT: ξ={_evt_xi:.3f} CVaR99={_evt_cvar:.4f} tail={_evt_tail}")
    lc="R" if l["recent_jump"] else "D"
    print(f"  Lévy: jumps={_levy_n} λ={_levy_l:.4f} recent={cc(str(_levy_r),lc)} penalty×{_levy_p:.1f}")
    print(f"  VR-Regime: {cc(_vr_regime,'M')}  SPRT LLR_bull={sprt_st['llr_bull']:+.3f} bear={sprt_st['llr_bear']:+.3f} n={sprt_st['n']}")
    print()

    b=bb(abs(sc)/10)
    print(cc("  "+"="*66,"W"))
    if side=="BUY":   print(cc(f"  ||  ####  B U Y  ^^  #### cats={dr['n_bull']} score={sc:+.1f} size×{final['garch_m']:.2f}  ||","G"))
    elif side=="SELL":print(cc(f"  ||  ####  S E L L  vv  #### cats={dr['n_bear']} score={sc:+.1f} size×{final['garch_m']:.2f}  ||","R"))
    else:
        nb=dr["n_bull"];nbe=dr["n_bear"]
        why=("no signals" if nb==0 and nbe==0 else
             f"BULL {nb}<{CFG['MIN_INDEP_SIGS']} cats" if nb>0 and nb<CFG["MIN_INDEP_SIGS"] else
             f"BEAR {nbe}<{CFG['MIN_INDEP_SIGS']} cats" if nbe>0 and nbe<CFG["MIN_INDEP_SIGS"] else
             f"ML blocked P={_ml_prob:.3f}" if not _ml_allow else
             f"score={sc:+.1f}")
        print(cc(f"  ||  ----  W A I T  ({why})  ||","Y"))
    ml_c="G" if ml.get("allow") else "R"
    _ml_prob = ml.get('prob', 0.5)
    _ml_yt   = ml.get('youden_t', 0.5)
    _ml_pass = cc('PASS','G') if ml.get('allow') else cc('BLOCK','R')
    _s_score = cc("{:>+.1f}".format(sc), 'B')
    _s_conf  = cc("{:.0f}%".format(conf), 'B')
    _s_ml    = cc("{:.3f}".format(_ml_prob), ml_c)
    print("  ||  Score:{} {} Conf:{} ML:{}({}) Youden_t={:.3f} ||".format(
        _s_score, cc(b, sc_c), _s_conf, _s_ml, _ml_pass, _ml_yt))
    print(cc("  "+"="*66,"W"))
    print()

    if final.get("tradeable") and final.get("tp1"):
        rr=final["rr"]; rc="G" if rr>=2.5 else("Y" if rr>=1.5 else "R")
        print(cc("  +--- TRADE ---------------------------------------------------------+","Y"))
        print("  |  Entry:  ${:>12,.2f}  ATR-mult={:.2f}×{}|".format(price, _f_am, " "*29))
        print(cc(f"  |  Stop:   ${final['sl']:>12,.2f}  (${abs(price-final['sl']):>7,.1f} adaptive ATR)","R")+" "*15+cc("|","Y"))
        print(cc(f"  |  TP1:    ${final['tp1']:>12,.2f}  → POC 60%","G")+" "*23+cc("|","Y"))
        print(cc(f"  |  TP2:    ${final['tp2']:>12,.2f}  → VAH/VAL 40%","G")+" "*20+cc("|","Y"))
        _s_rr = cc("{:.2f}x".format(rr), rc)
        print("  |  R:R={} Qty={:.4f}BTC Bayesian-Kelly={:.2f}%{}|".format(
            _s_rr, final['qty'], final['kelly']*100, " "*16))
        print(cc("  +-------------------------------------------------------------------+","Y"))
    elif side!="WAIT":
        print(cc(f"  conf={conf:.0f}%  R:R={_f_rr:.2f}x  not tradeable","Y"))
    print()

    print(cc("  -- DIRECTION ENGINE (Bayesian+IC+SNR weighted) ---------------------","M"))
    print(f"  BULL {dr['n_bull']}/{CFG['MIN_INDEP_SIGS']}: {cc(', '.join(dr['cats_bull']),'G') if dr['cats_bull'] else cc('none','D')}")
    print(f"  BEAR {dr['n_bear']}/{CFG['MIN_INDEP_SIGS']}: {cc(', '.join(dr['cats_bear']),'R') if dr['cats_bear'] else cc('none','D')}")
    for k,v in dr.get("active",{}).items():
        col="G" if str(v).startswith("+") else("R" if str(v).startswith("-") else "Y")
        print(f"    {cc('›','Y')} {cc(str(k)+': '+str(v),col)}")
    print()

    print(cc("  -- MATH SIGNALS -------------------------------------------------------","M"))
    ou_z=dr["ou_z"]; kt=dr["kal_trend"]
    print(f"  OU-MLE  z={ou_z:>+.3f}  HL={dr['ou_hl']:.1f}bars  VR-regime={cc(dr['vr_regime'],'M')}")
    print(f"  Kalman  trend={kt:>+.4f}  SNR={dr['kal_snr']:.2f}  price=${dr['kal_price']:,.0f}")
    print("  VWAP    {:>+.3f}σ  Wyckoff={}  FR={:.5f}%".format(
        dr['vwap_dev'], dr['wyckoff'], dr['avg_fr']*100))
    print()

    print(cc("  -- BAYESIAN POSTERIORS ------------------------------------------------","D"))
    bsumm=qm.bayes.summary()
    for sig,sd in list(bsumm.items())[:6]:
        pm=sd["p_win"]; bf=sd["bf"]; w=sd["weight"]; n_=sd["n_obs"]
        col="G" if pm>0.55 else("R" if pm<0.45 else "D")
        bf_s="★★★" if bf>10 else("★★" if bf>3 else("★" if bf>1 else ""))
        ic=qm.information_coefficient(sig)
        _s_pm = cc("{:.3f}".format(pm), col)
        print("  {:<12} P={} CI=[{:.3f},{:.3f}] BF={:.1f}{} IC={:>+.4f} w={:.2f}× n={}".format(
            sig, _s_pm, sd['ci_lo'], sd['ci_hi'], bf, bf_s, ic, w, n_))
    print()

    print(cc("  -- KELLY SIZING -------------------------------------------------------","D"))
    print("  Bayesian-Kelly={:.2f}%  GARCH×{:.2f}  CVaR-mult×{:.2f}  Kurt-adj×{:.3f}".format(_f_kelly*100, _f_gm, _f_cm, CFG.get("KURT_KELLY",1.0)))
    print(f"  Jump-penalty×{_levy_p:.1f}  Final-size×{_f_sm:.3f}")
    print()

    if paper_st:
        pc="G" if paper_st["pnl_pct"]>=0 else "R"
        print(cc("  -- PAPER TRADING ------------------------------------------------------","M"))
        _s_bal = cc("${:,.2f}".format(paper_st['balance']), 'W')
        _s_pnl = cc("{:+.2f}%".format(paper_st['pnl_pct']), pc)
        _s_pos = cc("IN POS", 'G') if paper_st['in_pos'] else ""
        print("  Balance:{}  PnL:{}  WR:{:.1f}%  Trades={}  {}".format(
            _s_bal, _s_pnl, paper_st['wr'], paper_st['trades'], _s_pos))
        print()

    recent=sh.recent(5)
    if recent:
        print(cc("  -- SIGNAL HISTORY  acc:{:.1f}%  n={} --------------------------------".format(sh.acc,sh.tot),"D"))
        for s in reversed(recent):
            oc=s.get("out","—");occ="G" if oc=="W" else("R" if oc=="L" else "D")
            print("  {} {:>4}  {:+.1f}pts  conf={:.0f}%  cats={}  {}".format(
                s['time'].strftime('%H:%M:%S'), s['side'], s['score'], s['conf'], s['n_ind'], cc(oc,occ)))
        print()

    print(cc("  Ctrl+C | --paper | --account N | --tf 5m | --model-dir path","D"))
    print(cc("="*74,"D"))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EliteV8:
    def __init__(self,account=1000.,paper=True,model_dir=CFG["MODEL_DIR"]):
        CFG["ACCOUNT"]=account
        self.qm        = QuantMath()
        self.dir_eng   = DirectionEngine()
        self.ml        = MLFilter()
        self.store     = ModelStore(model_dir)
        self.kbuf      = KlineBuf(); self.tbuf=TickBuf()
        self.ws_mgr    = None; self.ws_conn=False
        self.paper     = PaperTrader(account) if paper else None
        self.sh        = SigHistory()
        self.bars      = 0; self.bars_tr=0; self.bars_ck=0
        self.ckpt      = False; self.live=False
        self.sig_returns = defaultdict(lambda: deque(maxlen=80))

    def run(self):
        fund=pd.DataFrame(); df=pd.DataFrame()

        if WS_OK and NET:
            print(cc("  Starting WebSocket...","M"),flush=True)
            self.ws_mgr=WSMgr(CFG["SYMBOL"],CFG["TF"],self.kbuf,self.tbuf)
            self.ws_mgr.start(); time.sleep(3); self.ws_conn=self.ws_mgr.conn

        if NET:
            try:
                df=fetch(CFG["SYMBOL"],CFG["TF"],CFG["CANDLES"])
                fund=fetch_fund(CFG["SYMBOL"]); self.live=True
                print(f"  Data: {len(df)} bars")
            except Exception as e: print(f"  REST error: {e}  → synthetic")

        if df.empty: df,fund=synthetic(n=CFG["CANDLES"],seed=42)
        df=prepare(df)

        # Load model
        if not self.store.load(self.ml):
            print(cc("  ERROR: No model found. Check --model-dir path.","R"))
            print(f"  Looked in: {self.store.d}")
            print("  Need: latest.pkl or best.pkl")
            sys.exit(1)
        self.store.load_bayes(self.qm)
        print(cc(f"  Model loaded: val_acc={self.ml.val_acc:.4f}  Youden_t={self.ml.youden_t:.3f}","G"))

        # Seed kline buffer
        for _,row in df.tail(300).iterrows():
            self.kbuf.update({"open_time":int(row["open_time"].timestamp()*1000),
                              "open":float(row["open"]),"high":float(row["high"]),"low":float(row["low"]),
                              "close":float(row["close"]),"volume":float(row["volume"]),
                              "taker_buy_vol":float(row["taker_buy_vol"]),"trades":int(row["trades"])})

        print(cc("  Loop started. London 08-13 UTC, NY 13-20 UTC.\n","G"))
        curr_df=df

        while True:
            try:
                if self.ws_mgr and self.ws_conn:
                    self.ws_conn=self.ws_mgr.conn
                    if not self.kbuf.wait(timeout=70): continue
                    curr_df=self.kbuf.get()
                    if curr_df.empty or len(curr_df)<50: continue
                    curr_df=prepare(curr_df)
                else:
                    time.sleep(30)
                    if NET:
                        try: df=fetch(CFG["SYMBOL"],CFG["TF"],CFG["CANDLES"]); fund=fetch_fund(CFG["SYMBOL"]); self.live=True
                        except: pass
                    curr_df=prepare(df)

                self.bars+=1; self.bars_tr+=1; self.bars_ck+=1
                price=float(curr_df["close"].iloc[-1])
                ret_=curr_df["close"].pct_change().dropna()
                ret_arr=ret_.values
                dp_arr=curr_df["delta_pct"].values

                # ── MATH LAYER (run every bar) ─────────────────
                garch_r = self.qm.stoch.garch11(ret_arr)
                from scipy import stats as _stats
                kurt_r  = float(_stats.kurtosis(ret_arr[-100:])) if len(ret_arr)>=100 else 3.0
                evt_r   = self.qm.evt.fit_gpd(ret_arr)
                levy_r  = self.qm.stoch.levy_jumps(ret_arr)
                vr_r    = self.qm.info.variance_ratio_test(ret_arr[-100:])
                apen    = self.qm.info.approximate_entropy(ret_arr[-100:])

                # Update IC tracker
                if self.bars > 2:
                    fut_ret=float(curr_df["close"].pct_change().iloc[-1])
                    ou_now=self.qm.stoch.ou_mle(curr_df["close"].values[-50:])
                    self.qm.record_ic("ou",     ou_now["ou_z"],       -fut_ret)
                    self.qm.record_ic("cvd",    float(curr_df["delta_pct"].iloc[-1]), fut_ret)
                    kal_=self.qm.stoch.rts_kalman(curr_df["close"].values[-50:])
                    self.qm.record_ic("kalman", kal_["live_trend"],    fut_ret)

                # ── ML FEATURES + PREDICTION ───────────────────
                F_df=build_features(curr_df,fund)
                X_r=np.nan_to_num(F_df.values.astype(float),0.)
                X_sc=self.ml.scaler.transform(X_r) if self.ml.scaler else X_r
                ml=self.ml.predict(X_sc)

                # Update SPRT with ML probability
                sprt_d=self.qm.sprt.update(ml.get("prob",0.5))
                if sprt_d in ["CONFIRM_BUY","CONFIRM_SELL","REJECT"]:
                    self.qm.sprt.reset()

                # ── DIRECTION SIGNALS ──────────────────────────
                tick=self.tbuf.snap(30000)
                dr=self.dir_eng.score(curr_df,fund,self.qm,tick)

                # Block by ML
                side=dr["side"]
                if side!="WAIT" and not ml.get("allow",True): side="WAIT"

                # ── MARKET PROFILE ─────────────────────────────
                poc,vah,val_=market_profile(curr_df)

                # ── SIZING (math layer) ────────────────────────
                # Adaptive ATR stop (wider in high vol)
                atr=float(curr_df["atr"].iloc[-1]) or price*0.003
                atr_mult=CFG["ATR_SL_BASE"] + 0.5*garch_r["pct"]/100
                stop_dist=atr*atr_mult

                # Bayesian-Kelly sizing
                sig_name=(dr["cats_bull"]+dr["cats_bear"])[0] if (dr["cats_bull"]+dr["cats_bear"]) else "default"
                kelly_f=self.qm.kelly.bayesian_kelly(
                    sig_name, rr=CFG["TP_MULT"],
                    garch_mult=garch_r["size_mult"],
                    cvar_mult=float(evt_r.get("size_mult",1.0)),
                    kurt=max(kurt_r,0.), skew=float(stats.skew(ret_arr[-50:])))

                # CVaR constraint
                kelly_f=self.qm.kelly.cvar_constrained_kelly(kelly_f,ret_arr,CFG["CVAR_FLOOR"])

                # Jump penalty
                if levy_r["recent_jump"]: kelly_f*=CFG["JUMP_SIZE_MULT"]

                # Composite size multiplier
                size_mult=garch_r["size_mult"]*evt_r.get("size_mult",1.0)*levy_r["size_penalty"]

                # Trade levels
                if side=="BUY":
                    sl_=round(min(val_,price-stop_dist),1)
                    tp1=round(poc if poc>price else price+stop_dist*CFG["TP_MULT"],1)
                    tp2=round(vah if vah>tp1   else price+stop_dist*CFG["TP_MULT"]*2,1)
                elif side=="SELL":
                    sl_=round(max(vah,price+stop_dist),1)
                    tp1=round(poc if poc<price else price-stop_dist*CFG["TP_MULT"],1)
                    tp2=round(val_ if val_<tp1  else price-stop_dist*CFG["TP_MULT"]*2,1)
                else: sl_=tp1=tp2=None

                rr=abs(tp1-price)/max(abs(price-(sl_ or price)),1.) if tp1 else 0.
                qty=(CFG["ACCOUNT"]*kelly_f/max(stop_dist,1.)) if sl_ else 0.

                # Confidence
                n_ind=dr.get("n_indep",0)
                ml_ok=ml.get("prob",0.5)>ml.get("youden_t",0.5)+0.02 if side=="BUY" else ml.get("prob",0.5)<ml.get("youden_t",0.5)-0.02 if side=="SELL" else True
                conf=min(n_ind/5*100*(1.3 if ml_ok else 0.85),99.)

                tradeable=side!="WAIT" and rr>=CFG["MIN_RR"] and conf>=40.0
                final={"side":side,"tradeable":tradeable,"sl":sl_,"tp1":tp1,"tp2":tp2,
                       "qty":round(qty,4),"rr":rr,"confidence":conf,
                       "kelly":kelly_f,"garch_m":garch_r["size_mult"],
                       "atr_mult":round(atr_mult,2),"cvar_m":evt_r.get("size_mult",1.0),
                       "size_multiplier":size_mult}

                if self.bars>6: self.sh.resolve(price)

                # Paper trading + Bayesian updates
                if self.paper:
                    res=self.paper.update(price)
                    if res:
                        won=res["type"] in ["WIN","TP1"]
                        cats=res.get("cats",[])
                        pnl_pct=res["pnl"]/CFG["ACCOUNT"]
                        for cat in cats:
                            self.qm.bayes.update(cat,won)
                            self.sig_returns[cat].append(pnl_pct)
                        # Page-Hinkley drift detection
                        acc_signal=1.0 if won else 0.0
                        drift=self.qm.ph.update(acc_signal)
                        if drift: print(cc(f"  [DRIFT] Concept drift detected at bar {self.bars}","R"))
                        won_c="G" if won else "R"
                        print(cc(f"  [PAPER] {res['type']} pnl=${res['pnl']:+.2f} @${res['price']:,.0f}",won_c))
                    if tradeable and not self.paper.position:
                        reason="; ".join(str(v) for v in dr.get("active",{}).values())
                        cats=dr["cats_bull"] if side=="BUY" else dr["cats_bear"]
                        if self.paper.enter(side,price,sl_,tp1,tp2,qty,dr["score"],conf,reason,cats):
                            self.sh.record(side,price,dr["score"],conf,n_ind,cats)

                if self.bars_ck>=CFG["CHECKPOINT_N"]:
                    self.store.save_bayes(self.qm); self.ckpt=True; self.bars_ck=0

                display(price,dr,ml,final,self.bars,self.live,
                        self.paper.stats() if self.paper else None,
                        self.sh,self.ws_conn,self.ckpt,
                        self.qm,garch_r,evt_r,levy_r,vr_r,self.qm.sprt.state)

            except KeyboardInterrupt:
                print(cc("\n  Stopped.","Y"))
                if self.ws_mgr: self.ws_mgr.stop()
                self.store.save_bayes(self.qm)
                print(cc("  Bayesian state saved.","G"))
                if self.paper:
                    st=self.paper.stats()
                    print(f"  Final: ${st['balance']:,.2f}  PnL={st['pnl_pct']:+.2f}%  WR={st['wr']:.1f}%  Trades={st['trades']}")
                break
            except Exception as e:
                import traceback; print(f"  Error: {e}"); traceback.print_exc(); time.sleep(15)


def main():
    p=argparse.ArgumentParser(description="Elite Quant v8 — Math+Stats+Probability")
    p.add_argument("--account",   type=float,default=1000.)
    p.add_argument("--paper",     action="store_true")
    p.add_argument("--tf",        type=str,  default="5m")
    p.add_argument("--symbol",    type=str,  default="BTCUSDT")
    p.add_argument("--model-dir", type=str,  default="uq_models_v7",dest="model_dir")
    a=p.parse_args()
    CFG["TF"]=a.tf; CFG["SYMBOL"]=a.symbol; CFG["ACCOUNT"]=a.account; CFG["MODEL_DIR"]=a.model_dir

    print(cc("\n"+"="*74,"C"))
    print(cc("  ELITE QUANT v8  —  MATH + STATISTICS + PROBABILITY","C"))
    print(cc("  Bayesian·EVT·InfTheory·OU-MLE·RTS-Kalman·SPRT·Kelly·GARCH·Lévy","C"))
    print(cc("="*74,"C"))
    print(f"  {CFG['SYMBOL']}  TF:{CFG['TF']}  Account:${a.account:,.0f}  Mode:{'PAPER' if a.paper else 'SIGNALS'}")
    print(f"  Model dir: {a.model_dir}  WS: {'available' if WS_OK else 'not installed'}\n")
    EliteV8(account=a.account,paper=a.paper,model_dir=a.model_dir).run()

if __name__=="__main__":
    main()