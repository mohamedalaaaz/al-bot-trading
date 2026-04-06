#!/usr/bin/env python3
"""
QUANTUM ALPHA ENGINE v1.0 — Part 2: Orchestrator + Display
Run: python quantum_alpha.py
"""

import os, sys, time, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_alpha_p1 import *

import numpy as np
import pandas as pd
from scipy import stats, optimize
from collections import defaultdict
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import requests
    NET = True
except ImportError:
    NET = False

# ══════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════════════
def fetch_live():
    base = "https://fapi.binance.com"
    def kl(tf, n):
        r = requests.get(f"{base}/fapi/v1/klines",
                         params={"symbol":"BTCUSDT","interval":tf,"limit":n},
                         timeout=12)
        r.raise_for_status()
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","volume",
            "ct","qv","trades","taker_buy_vol","tbqv","_"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume","taker_buy_vol","trades"]:
            df[c] = df[c].astype(float)
        return df[["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]
    df_5m = kl("5m", 500)
    df_1h = kl("1h", 300)
    r2 = requests.get(f"{base}/fapi/v1/fundingRate",
                      params={"symbol":"BTCUSDT","limit":50}, timeout=10)
    r2.raise_for_status()
    fund = pd.DataFrame(r2.json())
    fund["fundingTime"] = pd.to_datetime(fund["fundingTime"], unit="ms", utc=True)
    fund["fundingRate"] = fund["fundingRate"].astype(float)
    return df_5m, df_1h, fund

def make_synthetic(seed=42):
    np.random.seed(seed)
    n = 500
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min", tz="UTC")
    price = 67200.0; rows = []
    for dt in dates:
        h = dt.hour; sv = 2.2 if h in [8,9,13,14,15,16] else 0.65
        mu = -0.00015 if h in [16,17,18] else 0.0001
        price = max(price*(1+np.random.normal(mu, 0.003*sv)), 50000)
        hi = price*(1+abs(np.random.normal(0, 0.0022*sv)))
        lo = price*(1-abs(np.random.normal(0, 0.0022*sv)))
        op = price*(1+np.random.normal(0, 0.001))
        vol = max(abs(np.random.normal(1200,420))*sv, 80)
        bsk = 0.64 if h in [8,9] else (0.36 if h in [17,18] else 0.50)
        tb = vol*np.clip(np.random.beta(bsk*7,(1-bsk)*7), 0.05, 0.95)
        if np.random.random() < 0.025: vol *= np.random.uniform(5,9)
        rows.append({"open_time":dt,"open":op,"high":hi,"low":lo,"close":price,
                     "volume":vol,"taker_buy_vol":tb,"trades":int(vol/0.04)})
    df = pd.DataFrame(rows)
    fr = pd.DataFrame([{"fundingTime":dates[i],
                        "fundingRate":np.random.normal(0.0001,0.0003)}
                       for i in range(0,n,96)])
    return df, fr

def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["body"]      = d["close"] - d["open"]
    d["body_pct"]  = d["body"] / d["open"] * 100
    d["is_bull"]   = d["body"] > 0
    d["wick_top"]  = d["high"] - d[["open","close"]].max(axis=1)
    d["wick_bot"]  = d[["open","close"]].min(axis=1) - d["low"]
    d["sell_vol"]  = d["volume"] - d["taker_buy_vol"]
    d["delta"]     = d["taker_buy_vol"] - d["sell_vol"]
    d["delta_pct"] = (d["delta"] / d["volume"].replace(0,np.nan)).fillna(0)
    hl  = d["high"] - d["low"]
    hpc = (d["high"] - d["close"].shift(1)).abs()
    lpc = (d["low"]  - d["close"].shift(1)).abs()
    d["atr"]   = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    rm = d["volume"].rolling(50).mean()
    rs = d["volume"].rolling(50).std()
    d["vol_z"] = (d["volume"] - rm) / rs.replace(0, np.nan)
    d["hour"]  = d["open_time"].dt.hour
    d["session"] = d["hour"].apply(
        lambda h: "Asia" if h<8 else "London" if h<13 else "NY" if h<20 else "Late")
    return d.fillna(0)


# ══════════════════════════════════════════════════════════════════════════
#  SIGNAL AGGREGATOR  (Module 9 + 10)
# ══════════════════════════════════════════════════════════════════════════
class SignalAggregator:
    """
    Combines ML ensemble + alpha scores + regime weighting
    → Final trade signal with conviction score.

    Three-tier confirmation:
    Tier 1: ML ensemble probability
    Tier 2: Alpha signal composite
    Tier 3: Execution timing + risk gate
    """

    @staticmethod
    def alpha_score(alphas: dict, regime_weights: dict) -> tuple:
        """Convert alpha dict to directional score, weighted by regime."""
        # Category → signals → direction
        signals = {
            # Momentum (positive = bull)
            "mom": sum([
                alphas.get("mom_mom_1",0)*2, alphas.get("mom_mom_3",0)*1.5,
                alphas.get("mom_mom_5",0), alphas.get("mom_macd",0)*500,
                alphas.get("mom_mom_acc",0)*100,
            ]) / 7,
            # Mean reversion (negative z = bull signal)
            "rev": sum([
                -alphas.get("rev_z_20",0), -alphas.get("rev_ou_z",0)*1.5,
                (50-alphas.get("rev_rsi_14",50))/50,    # rsi: below 50 = bull
                (50-alphas.get("rev_rsi_7",50))/50,
            ]) / 4,
            # Order flow (positive = bull)
            "of": sum([
                alphas.get("of_delta_pct",0)*2,
                alphas.get("of_div_bull",0)*3 - alphas.get("of_div_bear",0)*3,
                alphas.get("of_exhaust_sell",0)*2 - alphas.get("of_exhaust_buy",0)*2,
                alphas.get("of_vol_imb",0),
            ]) / 4,
            # Volume
            "vol": sum([
                -alphas.get("vol_vol_ratio",0)*0.5 + 0.5,  # <1 = vol contracting = bull
                alphas.get("vol_atr_ratio",0)*-0.3 + 0.3,
            ]) / 2,
            # Structure
            "struct": sum([
                alphas.get("struct_above_ema50",0.5) - 0.5,
                (alphas.get("struct_range_pos_20",0.5) - 0.5),
                alphas.get("struct_ema_slope",0) * 100,
            ]) / 3,
            # Microstructure
            "micro": sum([
                alphas.get("micro_wick_asym",0),
                alphas.get("micro_hl_position",0.5) - 0.5,
                alphas.get("micro_absorption",0)*(-1 if alphas.get("mom_mom_1",0)>0 else 1),
            ]) / 3,
            # Wyckoff
            "wyck": alphas.get("wyck_wyckoff_phase",0) / 3,
            # Funding (contrarian)
            "fund": alphas.get("fund_funding_revert",0),
        }

        # Apply regime weights
        rw = regime_weights
        weighted = {
            "momentum":   signals["mom"] * rw.get("momentum", 1.0),
            "mean_rev":   signals["rev"] * rw.get("mean_rev", 1.0),
            "order_flow": signals["of"]  * rw.get("order_flow",1.0),
            "volume":     signals["vol"] * rw.get("volume", 1.0),
            "structure":  signals["struct"],
            "micro":      signals["micro"],
            "wyckoff":    signals["wyck"],
            "funding":    signals["fund"],
        }

        total = sum(weighted.values())
        return float(np.clip(total, -5, 5)), weighted

    @staticmethod
    def combine(ml_score: int, ml_prob: float, alpha_raw: float,
                risk: dict, exec_q: dict, regime: dict) -> dict:
        """Final combination of all signals."""

        # ML gets extra weight when high confidence
        ml_w = 1.5 if ml_prob > 0.65 or ml_prob < 0.35 else 1.0

        # Alpha score normalized to similar scale
        alpha_sc = int(np.round(alpha_raw * 2.5))

        # Regime adjustment
        reg_mul = 1.2 if regime["name"] in ["QUIET_BULL","QUIET_BEAR"] else 0.8

        # Combined
        raw = (ml_score * ml_w + alpha_sc) * reg_mul

        # Risk gate: reduce size/signal if risk is elevated
        if not risk.get("go", True):
            return {"side":"WAIT","score":0,"confidence":0,
                    "tradeable":False,"reason":"Circuit breaker"}

        if risk.get("high_vol"):
            raw *= 0.6

        score = int(np.clip(raw, -12, 12))
        conf  = min(abs(score)/12*100, 99)
        conf  = conf * (1 - risk.get("vol_scale",1.0) * 0.1)

        side = "BUY" if score>=5 else "SELL" if score<=-5 else "WAIT"
        tradeable = (side!="WAIT" and conf>=55 and exec_q.get("enter_now",True))

        reasons = []
        if abs(ml_score) >= 2:
            reasons.append(f"{'+'if ml_score>0 else''}{int(ml_score*ml_w):.0f} ML(p={ml_prob:.3f})")
        if abs(alpha_sc) >= 2:
            reasons.append(f"{'+'if alpha_sc>0 else''}{alpha_sc} Alphas")
        if regime["name"] != "QUIET_BULL":
            reasons.append(f"Regime:{regime['name']}")

        return {
            "side":      side,
            "score":     score,
            "confidence":float(conf),
            "tradeable": tradeable,
            "ml_score":  ml_score,
            "alpha_raw": alpha_raw,
            "reasons":   reasons,
        }


# ══════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════
ANSI = {"G":"\033[92m","R":"\033[91m","Y":"\033[93m","C":"\033[96m",
        "W":"\033[97m","B":"\033[1m","D":"\033[2m","M":"\033[95m",
        "K":"\033[90m","X":"\033[0m"}
def c(t, col): return f"{ANSI.get(col,'')}{t}{ANSI['X']}"
def bar(v, w=12, ch="█", em="░"):
    f=min(int(abs(v)*w),w); return ch*f+em*(w-f)

def display(price, final, ml_res, alpha_weighted, regime, risk,
            exec_q, bl_size, feat_imp, live, loop_n, train_res):
    os.system("cls" if os.name=="nt" else "clear")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    side = final["side"]
    sc   = final["score"]
    conf = final["confidence"]
    sc_col = "G" if sc>0 else ("R" if sc<0 else "Y")
    sd_col = "G" if side=="BUY" else ("R" if side=="SELL" else "Y")

    # ── Header ──
    print(c("╔"+"═"*72+"╗","C"))
    print(c("║         QUANTUM ALPHA ENGINE  v1.0  │  BTC/USDT FUTURES                ║","C"))
    print(c("╚"+"═"*72+"╝","C"))
    print(f"  {c(now,'D')}   Loop #{loop_n}   "
          f"{'🟢 LIVE' if live else c('🟡 SYNTHETIC','Y')}")
    print(f"  {c(f'Price: ${price:>12,.2f}','W')}   "
          f"ATR: ${risk.get('vol20',0)*price*288**0.5:,.1f}   "
          f"Regime: {c(regime['name'], 'M')}")
    print()

    # ── BIG SIGNAL ──
    blen = min(abs(sc), 12)
    b_bar = "█"*blen + "░"*(12-blen)
    print(c("  ┌"+"─"*68+"┐","W"))
    if side=="BUY":
        print(c("  │  ╔══════════════════════════════════════════════════════════╗  │","G"))
        print(c("  │  ║          ██████  B U Y  ▲▲▲▲▲▲▲▲▲  ██████             ║  │","G"))
        print(c("  │  ╚══════════════════════════════════════════════════════════╝  │","G"))
    elif side=="SELL":
        print(c("  │  ╔══════════════════════════════════════════════════════════╗  │","R"))
        print(c("  │  ║          ██████  S E L L  ▼▼▼▼▼▼▼▼  ██████            ║  │","R"))
        print(c("  │  ╚══════════════════════════════════════════════════════════╝  │","R"))
    else:
        print(c("  │  ─────────────  W A I T  (insufficient confluence) ─────────  │","Y"))
    print(f"  {c('│','W')}  Score: {c(f'{sc:>+3d}','B')}  {c(b_bar, sc_col)}  "
          f"Confidence: {c(f'{conf:.1f}%','B')}   Regime×{regime['weights'].get('order_flow',1):.1f}  {c('│','W')}")
    print(c("  └"+"─"*68+"┘","W"))
    print()

    # ── Trade Setup ──
    if final.get("tradeable") and final.get("tp1"):
        rr = final.get("rr", 0)
        rrc = "G" if rr>=2.0 else ("Y" if rr>=1.5 else "R")
        print(c("  ┌── OPTIMAL TRADE STRUCTURE (Execution IQ) ──────────────────────┐","Y"))
        print(c("  │","Y")+f"  Entry:  ${price:>12,.2f}  ({exec_q.get('exec_type','MARKET')} order)"+" "*24+c("│","Y"))
        print(c("  │","Y")+c(f"  Stop:   ${final['sl']:>12,.2f}  "
              f"(${abs(price-final['sl']):>7,.1f} below structure/ATR)","R")+" "*8+c("│","Y"))
        print(c("  │","Y")+c(f"  TP1:    ${final['tp1']:>12,.2f}  close 60% → POC/structure","G")+" "*12+c("│","Y"))
        print(c("  │","Y")+c(f"  TP2:    ${final['tp2']:>12,.2f}  close 40% → VAH/VAL","G")+" "*17+c("│","Y"))
        print(c("  │","Y")+f"  R:R={c(f'{rr:.2f}x',rrc)}  "
              f"Qty={final.get('qty',0):.3f} BTC  "
              f"Risk=${bl_size[0]:.2f}  BL-Kelly={bl_size[1]*100:.2f}%"+" "*8+c("│","Y"))
        vdev = exec_q.get("vwap_dev",0)
        eq_score = exec_q.get('exec_score',0)
        eq_str = c(f'{eq_score}/4', 'B')
        vdev2 = exec_q.get('vwap_dev',0)
        klam = exec_q.get('kyle_lambda',0)
        print(c('  │','Y')+f'  Exec quality: {eq_str}  '
              f'VWAP dev={vdev2:>+.4f}  '
              f'Impact λ={klam:.5f}'+' '*8+c('│','Y'))
        print(c("  └"+"─"*68+"┘","Y"))
    elif side != "WAIT":
        print(c(f"  No trade: conf={conf:.1f}% or exec_quality={exec_q.get('exec_score',0)}/4 insufficient","Y"))
    print()

    # ── ML Ensemble Panel ──
    mp = ml_res.get("model_probs", {"gbm":0.5,"rf":0.5,"nn":0.5})
    p  = ml_res.get("prob",0.5)
    nn_c = "G" if p>0.56 else ("R" if p<0.44 else "Y")
    print(c("  ── ML ENSEMBLE (GBM + RF + NN + Meta-Learner) ────────────────────","M"))
    print(f"  {'Architecture:':<22} GBM(200) + RF(200) + NN(64→32→16) → Logistic Meta")
    vacc = ml_res.get("val_acc",0)*100
    print(f"  {'Val accuracy:':<22} {c(f'{vacc:.2f}%','B')}")
    print(f"  {'GBM → RF → NN:':<22} [{mp.get('gbm',0.5):.4f},  {mp.get('rf',0.5):.4f},  {mp.get('nn',0.5):.4f}]  "
          f"std={ml_res.get('std',0):.4f}")
    msc = ml_res.get("score",0)
    print(f"  {'Meta P(UP):':<22} {c(f'{p:.4f}', nn_c)}  ML score: {c(f'{msc:>+d}','B')}")
    print()

    # ── Alpha Signal Scorecard ──
    print(c("  ── ALPHA FACTORY (80+ signals → 8 categories) ───────────────────","M"))
    cats = [
        ("Momentum",    alpha_weighted.get("momentum",0),    "follows trend"),
        ("Mean Rev",    alpha_weighted.get("mean_rev",0),     "fades extremes"),
        ("Order Flow",  alpha_weighted.get("order_flow",0),   "institutional flow"),
        ("Volume",      alpha_weighted.get("volume",0),       "volume regime"),
        ("Structure",   alpha_weighted.get("structure",0),    "VWAP/levels"),
        ("Microstr",    alpha_weighted.get("micro",0),        "wick/efficiency"),
        ("Wyckoff",     alpha_weighted.get("wyckoff",0),      "smart money cycle"),
        ("Funding",     alpha_weighted.get("funding",0),      "contrarian FR"),
    ]
    for name, val, desc in cats:
        col  = "G" if val>0.05 else ("R" if val<-0.05 else "D")
        blen = min(int(abs(val)*8),8)
        b    = ("█"*blen+"░"*(8-blen)) if blen>0 else "─"*8
        sign = "+" if val>=0 else ""
        print(f"  {name:<12} {c(f'{sign}{val:.3f}','B')} {c(b,col)}  {c(desc,'D')}")
    print()

    # ── HMM Regime ──
    print(c("  ── HIDDEN MARKOV MODEL REGIME ──────────────────────────────────","M"))
    reg_col = "G" if "BULL" in regime["name"] else "R"
    print(f"  Current regime:  {c(regime['name'], reg_col)}")
    print(f"  Signal weights:  " +
          "  ".join([f"{k}×{v:.1f}" for k,v in regime["weights"].items()]))
    print()

    # ── Risk Engine ──
    print(c("  ── RISK ENGINE (Black-Litterman + CVaR + Vol Scaling) ────────────","M"))
    go_str = c("GO ✓","G") if risk.get("go") else c("STOP ✗","R")
    vc = "G" if risk.get("vol_scale",1)>=1.0 else "Y"
    print(f"  Risk gate:       {go_str}   "
          f"Vol regime: {'HIGH ⚠' if risk.get('high_vol') else 'NORMAL'}")
    print(f"  CVaR(95%):       {risk.get('cvar_95',0)*100:>+.4f}%")
    vs_val = risk.get("vol_scale",1)
    print(f"  Vol scale:       {c(f'{vs_val:.2f}x',vc)}  "
          f"(adjusts size by realized vol)")
    print(f"  Adj risk:        {risk.get('adj_risk_pct',0.01)*100:.2f}% of account")
    print()

    # ── Top Alpha Features (GBM importance) ──
    if feat_imp:
        print(c("  ── TOP PREDICTIVE ALPHAS (GBM importance) ───────────────────────","M"))
        for i,(k,v) in enumerate(list(feat_imp.items())[:8]):
            blen = min(int(v*400),20)
            b = "█"*blen
            print(f"  {i+1:>2}. {k:<30} {b}  {v:.4f}")
        print()

    # ── Training Summary ──
    print(c("  ── MODEL TRAINING SUMMARY ───────────────────────────────────────","D"))
    print(f"  {'GBM val acc:':<20} {train_res.get('gbm_acc',0)*100:.2f}%")
    print(f"  {'RF val acc:':<20} {train_res.get('rf_acc',0)*100:.2f}%")
    print(f"  {'NN val acc:':<20} {train_res.get('nn_acc',0)*100:.2f}%")
    print(f"  {'Meta val acc:':<20} {ml_res.get('val_acc',0)*100:.2f}%")
    print(f"  {'Alphas built:':<20} {train_res.get('n_alphas',0)}")
    print(f"  {'Training samples:':<20} {train_res.get('n_samples',0)}")
    print()

    # ── Reasons ──
    print(c("  ── CONFLUENCE REASONS ─────────────────────────────────────────────","D"))
    reasons = final.get("reasons", [])
    if reasons:
        print("  " + "  │  ".join(reasons))
    else:
        print("  No strong individual signal dominates — composite edge only")

    print()
    print(c("  Ctrl+C to stop  │  --loop for continuous  │  --account N for custom size","D"))
    print(c("═"*74,"D"))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════
class QuantumAlphaEngine:
    def __init__(self, account=1000.0, max_risk=0.015):
        self.account    = account
        self.max_risk   = max_risk
        self.alpha_fac  = AlphaFactory()
        self.ml         = MLEnsemble()
        self.hmm        = HiddenMarkovRegime()
        self.bl         = BlackLittermanSizer()
        self.risk_eng   = RiskEngine()
        self.exec_eng   = ExecutionEngine()
        self.aggregator = SignalAggregator()
        self.loop_n     = 0
        self.trained    = False
        self.train_res  = {}

    def train(self, df: pd.DataFrame, funding: pd.DataFrame, verbose=True):
        """Full training pipeline."""
        if verbose:
            print(c("\n  ── TRAINING QUANTUM ALPHA ENGINE ────────────────────────────","M"))
            print(f"  Building alpha matrix from {len(df)} bars...")

        X, y, feat_names = self.alpha_fac.build_matrix(df, funding, target_bars=3)

        if len(X) < 100:
            print("  Insufficient data"); return

        n_alphas = X.shape[1]
        if verbose:
            print(f"  Alpha matrix: {X.shape[0]} samples × {n_alphas} features")
            print(f"  Bull rate: {y.mean()*100:.1f}%  Bear rate: {(1-y.mean())*100:.1f}%")

        if verbose: print(f"\n  Training ML ensemble (stacked generalization)...")
        self.ml.train(X, y, feat_names, verbose=verbose)

        self.trained   = True
        self.train_res = {
            "n_alphas":  n_alphas,
            "n_samples": len(X),
            "gbm_acc":   float(((self.ml.gbm.predict_proba(
                             self.ml.scaler_gbm.transform(X[-30:]))[:,1]>0.5
                         ).astype(int) == y[-30:]).mean()),
            "rf_acc":    float(((self.ml.rf.predict_proba(
                             self.ml.scaler_gbm.transform(X[-30:]))[:,1]>0.5
                         ).astype(int) == y[-30:]).mean()),
            "nn_acc":    float(self.ml.val_score),
        }
        if verbose:
            print(f"\n  Training complete. Meta val_acc={self.ml.val_score:.4f}")

    def run_once(self) -> dict:
        self.loop_n += 1

        # ── Get data ──
        live = False
        if NET:
            try:
                df_p, df_h, funding = fetch_live()
                live = True
            except Exception as e:
                df_p, funding = make_synthetic(seed=self.loop_n%10)
        else:
            df_p, funding = make_synthetic(seed=self.loop_n%10)

        df_p = prep(df_p)
        price = float(df_p["close"].iloc[-1])
        atr   = float(df_p["atr"].iloc[-1]) or price*0.003

        # ── Train if needed ──
        if not self.trained:
            self.train(df_p, funding, verbose=True)

        # ── Current alphas ──
        alphas = self.alpha_fac.build_all(df_p, funding)

        # ── ML prediction ──
        X, _, feat_names = self.alpha_fac.build_matrix(df_p, funding, target_bars=3)
        if len(X) > 0 and self.trained:
            ml_res = self.ml.predict(X)
        else:
            ml_res = {"prob":0.5,"score":0,"model_probs":{},"val_acc":0,"std":0}

        # ── HMM Regime ──
        regime = self.hmm.current_regime(df_p)

        # ── Alpha → score ──
        alpha_raw, alpha_weighted = self.aggregator.alpha_score(alphas, regime["weights"])

        # ── Risk assessment ──
        risk = self.risk_eng.assess(df_p, self.max_risk, self.account)

        # ── Market profile for structure ──
        c_  = df_p["close"].astype(float)
        lo, hi = df_p["low"].min(), df_p["high"].max()
        tick = max((hi-lo)/40, 10.0)
        bkts = np.arange(np.floor(lo/tick)*tick, np.ceil(hi/tick)*tick+tick, tick)
        vm   = defaultdict(float)
        for _, row in df_p.iterrows():
            lvls = bkts[(bkts>=row["low"])&(bkts<=row["high"])]
            if not len(lvls): continue
            vp = row["volume"]/len(lvls)
            for lv in lvls: vm[lv] += vp
        poc = vah = val = price
        if vm:
            pf  = pd.DataFrame({"p":list(vm.keys()),"v":list(vm.values())}).sort_values("p")
            poc = float(pf.loc[pf["v"].idxmax(),"p"])
            tot = pf["v"].sum(); pi=pf["v"].idxmax(); cum=0; va=[]
            while cum/tot<0.70:
                ui,li = pi+1,pi-1
                uv = pf.loc[ui,"v"] if ui in pf.index else 0
                dv = pf.loc[li,"v"] if li in pf.index else 0
                if uv>=dv and ui in pf.index: va.append(ui); cum+=uv; pi=ui
                elif li in pf.index: va.append(li); cum+=dv; pi=li
                else: break
            if va:
                vah=float(pf.loc[va,"p"].max()); val=float(pf.loc[va,"p"].min())

        # ── Final signal ──
        side_prelim = "BUY" if (ml_res.get("score",0) + alpha_raw*2.5) > 3 else \
                      "SELL" if (ml_res.get("score",0) + alpha_raw*2.5) < -3 else "WAIT"
        exec_q  = self.exec_eng.execution_score(df_p, side_prelim)
        levels  = self.exec_eng.optimal_levels(price, atr, side_prelim, poc, vah, val)
        final   = self.aggregator.combine(
            ml_res.get("score",0), ml_res.get("prob",0.5),
            alpha_raw, risk, exec_q, regime
        )

        # ── Sizing ──
        rr     = levels["rr"]
        bl_sz  = self.bl.optimal_size(
            ml_res.get("prob",0.5), rr, final["confidence"],
            self.account, risk.get("adj_risk_pct", 0.01)
        )
        stop_dist = abs(price - levels["stop"]) if levels["stop"] else atr*1.5
        qty = bl_sz[0] / max(stop_dist * 5, 1)   # leveraged

        final.update({
            "sl":  levels["stop"],
            "tp1": levels["tp1"],
            "tp2": levels["tp2"],
            "qty": round(qty, 3),
            "rr":  rr,
        })

        display(price, final, ml_res, alpha_weighted, regime, risk,
                exec_q, bl_sz, self.ml.feat_imp, live, self.loop_n, self.train_res)

        return final


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Quantum Alpha Engine — Institutional BTC Trading")
    p.add_argument("--loop",    action="store_true", help="Run continuously")
    p.add_argument("--interval",type=int, default=30, help="Loop interval seconds")
    p.add_argument("--account", type=float,default=1000.0, help="Account size USDT")
    p.add_argument("--risk",    type=float,default=0.015,  help="Max risk per trade")
    p.add_argument("--retrain", type=int,  default=10,     help="Retrain every N loops")
    args = p.parse_args()

    print(c("\n" + "▓"*74, "C"))
    print(c("  QUANTUM ALPHA ENGINE  v1.0  —  Institutional BTC/USDT Futures", "C"))
    print(c("  GBM + RF + NN + Meta | 80+ Alphas | HMM Regime | Black-Litterman", "C"))
    print(c("▓"*74, "C"))
    print(f"\n  Account:  ${args.account:>10,.2f} USDT")
    print(f"  Max risk: {args.risk*100:.1f}% per trade")
    print(f"  Mode:     {'LIVE LOOP' if args.loop else 'SINGLE RUN'}")

    engine = QuantumAlphaEngine(account=args.account, max_risk=args.risk)

    if args.loop:
        n = 0
        while True:
            try:
                n += 1
                if n % args.retrain == 1:
                    engine.trained = False   # force retrain
                engine.run_once()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print(c("\n  Stopped. Good trading.", "Y")); break
            except Exception as e:
                print(f"  Error: {e}"); import traceback; traceback.print_exc()
                time.sleep(15)
    else:
        engine.run_once()


if __name__ == "__main__":
    main()
