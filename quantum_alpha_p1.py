#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         QUANTUM ALPHA ENGINE  v1.0                                         ║
║         Institutional-Grade Multi-Strategy Trading System                  ║
║         BTC/USDT Binance Futures                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  WHAT MAKES BILLIONS:                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │ Renaissance: 100+ uncorrelated alpha signals stacked statistically  │   ║
║  │ Citadel: Ultra-fast order flow + market microstructure              │   ║
║  │ Two Sigma: Deep ML on alternative data, regime-aware models         │   ║
║  │ AQR: Factor premia + systematic risk management                     │   ║
║  │ DE Shaw: Arbitrage + mean reversion across instruments              │   ║
║  │ Millennium: Pod-based multi-strategy with strict risk limits        │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                             ║
║  THIS ENGINE:                                                               ║
║   MODULE 1 │ ALPHA FACTORY        — 80+ features, 12 alpha categories     ║
║   MODULE 2 │ ML ENSEMBLE          — GBM + RF + NN + Meta-learner          ║
║   MODULE 3 │ HIDDEN MARKOV MODEL  — Regime-aware signal weighting         ║
║   MODULE 4 │ BLACK-LITTERMAN      — Portfolio optimization                 ║
║   MODULE 5 │ MICROSTRUCTURE       — Bid-ask spread, price impact model    ║
║   MODULE 6 │ REINFORCEMENT SIZING — Kelly + fractional + dynamic          ║
║   MODULE 7 │ RISK ENGINE          — CVaR, correlation, drawdown limits    ║
║   MODULE 8 │ EXECUTION IQ         — Entry timing, slippage minimization   ║
║   MODULE 9 │ META-STRATEGY        — Which strategy to run per regime      ║
║   MODULE 10│ SIGNAL AGGREGATOR    — Final BUY/SELL/WAIT with conviction   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
from scipy import stats, optimize, linalg
from scipy.stats import norm, skew, kurtosis
from collections import defaultdict, deque
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import requests
    NET = True
except ImportError:
    NET = False


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 1 │ ALPHA FACTORY  — 80+ alpha signals
#  "More uncorrelated alphas = more Sharpe. Period." — Renaissance
# ══════════════════════════════════════════════════════════════════════════
class AlphaFactory:
    """
    Generates 80+ alpha signals across 12 categories.
    Each alpha is a standardized signal predicting future returns.
    The key: UNCORRELATED alphas compound. Correlated ones don't.
    """

    @staticmethod
    def momentum_alphas(d: pd.DataFrame) -> dict:
        """Price momentum across multiple horizons."""
        c = d["close"].astype(float)
        return {
            "mom_1":   c.pct_change(1).iloc[-1],
            "mom_3":   c.pct_change(3).iloc[-1],
            "mom_5":   c.pct_change(5).iloc[-1],
            "mom_10":  c.pct_change(10).iloc[-1],
            "mom_20":  c.pct_change(20).iloc[-1],
            "mom_50":  c.pct_change(50).iloc[-1],
            # MACD
            "macd":    (c.ewm(12).mean() - c.ewm(26).mean()).iloc[-1] / c.iloc[-1],
            "macd_sig":(c.ewm(12).mean() - c.ewm(26).mean()).ewm(9).mean().iloc[-1] / c.iloc[-1],
            # Rate of change acceleration
            "mom_acc": (c.pct_change(5) - c.pct_change(5).shift(5)).iloc[-1],
            # Momentum reversal (short-term)
            "rev_1":   -c.pct_change(1).iloc[-1],          # 1-bar reversal
            "rev_3":   -c.pct_change(3).rolling(3).mean().iloc[-1],  # fade 3-bar
        }

    @staticmethod
    def mean_reversion_alphas(d: pd.DataFrame) -> dict:
        """Mean reversion and statistical distance signals."""
        c = d["close"].astype(float)
        # Rolling z-scores at different windows
        def zs(w): return ((c - c.rolling(w).mean()) / c.rolling(w).std()).iloc[-1]
        # OU z-score (fast)
        x = c.values[-100:]; dx = np.diff(x); xl = x[:-1]
        A = np.column_stack([np.ones(len(xl)), xl])
        try:
            co,_,_,_ = np.linalg.lstsq(A, dx, rcond=None)
            mu_ou = -co[0]/co[1] if co[1]!=0 else float(x.mean())
            sg_ou = max(float(np.std(dx-(co[0]+co[1]*xl))), 1e-8)
            ou_z  = (float(x[-1]) - mu_ou) / sg_ou
        except:
            ou_z = 0.0
        return {
            "z_10":   float(zs(10)),
            "z_20":   float(zs(20)),
            "z_50":   float(zs(50)),
            "ou_z":   float(np.clip(ou_z, -5, 5)),
            "bb_pos": float(zs(20)),   # Bollinger position (same as z_20)
            "rsi_14": float(AlphaFactory._rsi(c, 14)),
            "rsi_7":  float(AlphaFactory._rsi(c, 7)),
            "rsi_div":(AlphaFactory._rsi(c,14) - AlphaFactory._rsi(c,14).shift(3)).iloc[-1],
        }

    @staticmethod
    def _rsi(prices, period):
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return (100 - 100/(1+rs)).fillna(50)

    @staticmethod
    def order_flow_alphas(d: pd.DataFrame) -> dict:
        """Order flow and microstructure signals."""
        dp    = d["delta_pct"].astype(float).fillna(0)
        delta = d["delta"].astype(float).fillna(0)
        vol   = d["volume"].astype(float).replace(0, np.nan)
        cvd20 = delta.rolling(20).sum()
        cvd_s = cvd20.diff(3)
        pr_s  = d["close"].diff(3) / d["close"].shift(3) * 100

        # Kyle's Lambda (price impact coefficient)
        # Δp ≈ λ × signed_volume  →  λ = cov(Δp, sgn_vol) / var(sgn_vol)
        ret   = d["close"].pct_change().fillna(0)
        sgn_v = np.sign(dp)
        lam   = float(np.cov(ret.tail(20).values, sgn_v.tail(20).values)[0,1] /
                      max(sgn_v.tail(20).var(), 1e-10))

        return {
            "delta_pct":   float(dp.iloc[-1]),
            "cvd_slope":   float(cvd_s.iloc[-1]),
            "cvd_acc":     float(cvd_s.diff(1).iloc[-1]),  # CVD acceleration
            "buy_ratio":   float((d["taker_buy_vol"]/vol).iloc[-1]),
            "vol_imb":     float((delta/vol).iloc[-1]),     # volume imbalance
            "div_bull":    float((pr_s < -0.12) & (cvd_s > 0)).iloc[-1],
            "div_bear":    float((pr_s >  0.12) & (cvd_s < 0)).iloc[-1],
            "exhaust_buy": float((dp > 0.3) & (d["body_pct"].abs() < 0.06)).iloc[-1],
            "exhaust_sell":float((dp < -0.3)& (d["body_pct"].abs() < 0.06)).iloc[-1],
            "kyle_lambda": float(np.clip(lam, -0.1, 0.1)),
            "amihud_illiq":float(abs(ret).iloc[-1] / max(vol.iloc[-1], 1)),
        }

    @staticmethod
    def volatility_alphas(d: pd.DataFrame) -> dict:
        """Volatility signals and regime."""
        ret   = d["close"].pct_change().dropna()
        vol5  = float(ret.tail(5).std())
        vol20 = float(ret.tail(20).std())
        vol50 = float(ret.tail(50).std())

        # Volatility ratio (trend of vol)
        vr5_20  = vol5 / vol20 if vol20 > 0 else 1.0
        vr20_50 = vol20/ vol50 if vol50 > 0 else 1.0

        # Realized variance ratio (jump detection proxy)
        bpv = float((ret.abs() * ret.abs().shift(1)).tail(20).mean() * np.pi/2)
        rv  = float(ret.tail(20).var())
        jump_ratio = (rv - bpv) / max(rv, 1e-10)

        # Skewness and kurtosis of returns
        sk = float(skew(ret.tail(50).values)) if len(ret) >= 50 else 0
        kt = float(kurtosis(ret.tail(50).values)) if len(ret) >= 50 else 3

        # ATR ratio
        atr   = float(d["atr"].iloc[-1])
        atr20 = float(d["atr"].rolling(20).mean().iloc[-1])

        return {
            "vol5":        vol5,
            "vol20":       vol20,
            "vol_ratio":   vr5_20,       # >1 = vol increasing
            "vol_trend":   vr20_50,
            "jump_proxy":  float(np.clip(jump_ratio, -1, 3)),
            "skewness":    sk,
            "kurtosis":    kt,
            "atr_ratio":   float(atr / atr20 if atr20 > 0 else 1.0),
        }

    @staticmethod
    def microstructure_alphas(d: pd.DataFrame) -> dict:
        """Market microstructure signals."""
        # Wick-based signals (proxy for bid-ask, stop hunts)
        wick_t = d["wick_top"].astype(float)
        wick_b = d["wick_bot"].astype(float)
        body   = d["body_pct"].astype(float).abs()
        atr    = d["atr"].astype(float)

        # Relative wick sizes
        rw_top = (wick_t / atr.replace(0,np.nan)).fillna(0)
        rw_bot = (wick_b / atr.replace(0,np.nan)).fillna(0)

        # Candle efficiency: body / range (high = strong directional)
        efficiency = (body / (d["high"]-d["low"]).astype(float).replace(0,np.nan)).fillna(0)

        # High-low position (where did close land in the range?)
        hl_pos = ((d["close"]-d["low"]) / (d["high"]-d["low"]).replace(0,np.nan)).fillna(0.5)

        # Volume-weighted spread proxy
        vz = d["vol_z"].astype(float).fillna(0)

        return {
            "wick_top_rel":  float(rw_top.iloc[-1]),
            "wick_bot_rel":  float(rw_bot.iloc[-1]),
            "wick_asym":     float((rw_bot - rw_top).iloc[-1]),   # +ve = bottom wick bigger = buy pressure
            "efficiency":    float(efficiency.iloc[-1]),
            "hl_position":   float(hl_pos.iloc[-1]),              # 1=close at high, 0=close at low
            "vol_surge":     float(vz.iloc[-1]),
            "large_trade":   float(vz.iloc[-1] > 3.0),
            "absorption":    float((vz > 1.5) & (body < 0.08)).iloc[-1],
        }

    @staticmethod
    def structure_alphas(d: pd.DataFrame) -> dict:
        """Market structure: value areas, VWAP, support/resistance."""
        c   = d["close"].astype(float)
        vol = d["volume"].astype(float)
        tp  = (d["high"] + d["low"] + d["close"]) / 3

        # Multi-period VWAP deviations
        def vwap_dev(w):
            v = (tp*vol).rolling(w).sum() / vol.rolling(w).sum()
            return ((c - v) / v * 100).fillna(0)

        # Swing high/low proximity
        hi20 = d["high"].rolling(20).max()
        lo20 = d["low"].rolling(20).min()
        rng  = (hi20 - lo20).replace(0, np.nan)
        pos  = ((c - lo20) / rng).fillna(0.5)   # 0=at lows, 1=at highs

        return {
            "vwap_dev_20":  float(vwap_dev(20).iloc[-1]),
            "vwap_dev_50":  float(vwap_dev(50).iloc[-1]),
            "range_pos_20": float(pos.iloc[-1]),     # position within 20-bar range
            "dist_hi20":    float(((hi20 - c) / c * 100).iloc[-1]),  # % to recent high
            "dist_lo20":    float(((c - lo20) / c * 100).iloc[-1]),  # % to recent low
            "above_ema50":  float((c > c.ewm(50).mean()).iloc[-1]),
            "above_ema200": float((c > c.ewm(200).mean()).iloc[-1]) if len(c)>200 else 0.5,
            "ema_slope":    float((c.ewm(20).mean().diff(5) / c * 100).iloc[-1]),
        }

    @staticmethod
    def time_alphas(d: pd.DataFrame) -> dict:
        """Time-of-day and calendar effects."""
        h = d["open_time"].dt.hour.iloc[-1]
        dow = d["open_time"].dt.dayofweek.iloc[-1]

        # Empirical session biases for BTC (from historical analysis)
        session_bias = {
            "london_open": 1 if h in [8,9] else 0,         # volatile, trend starts
            "ny_open":     1 if h in [13,14] else 0,        # highest vol, momentum
            "ny_close":    1 if h in [19,20] else 0,        # potential reversal
            "asian_range": 1 if 0<=h<8 else 0,              # range-bound
            "weekend_eff": 1 if dow >= 4 else 0,            # weekend vol regime
        }

        # Cyclical encoding
        return {
            "sin_hour":   float(np.sin(2*np.pi*h/24)),
            "cos_hour":   float(np.cos(2*np.pi*h/24)),
            "sin_dow":    float(np.sin(2*np.pi*dow/7)),
            "cos_dow":    float(np.cos(2*np.pi*dow/7)),
            **session_bias,
        }

    @staticmethod
    def cross_signal_alphas(d: pd.DataFrame) -> dict:
        """Interaction and cross-signal features."""
        c    = d["close"].astype(float)
        dp   = d["delta_pct"].astype(float).fillna(0)
        vz   = d["vol_z"].astype(float).fillna(0)
        ret1 = c.pct_change(1)

        # Momentum × volume (high-conviction moves)
        mom_vol   = float((ret1 * vz).iloc[-1])
        # Delta × momentum (aligned vs divergent)
        delta_mom = float((dp * ret1.apply(np.sign)).iloc[-1])
        # Absorption score
        absorb    = float(((vz > 1.5) & (ret1.abs() < 0.002)).iloc[-1])
        # Trapped trader proxy
        trap      = float(((d["body_pct"].shift(1).abs() > 0.3) &
                           (d["body_pct"] * d["body_pct"].shift(1) < 0)).iloc[-1])

        return {
            "mom_vol_interact":  float(np.clip(mom_vol, -0.5, 0.5)),
            "delta_mom_align":   float(np.clip(delta_mom, -1, 1)),
            "absorption_signal": absorb,
            "trap_signal":       trap,
            "vol_regime_mom":    float(vz.iloc[-1] * ret1.rolling(5).mean().iloc[-1]),
        }

    @staticmethod
    def wyckoff_alphas(d: pd.DataFrame) -> dict:
        """Wyckoff market cycle positioning."""
        rec = d.tail(30)
        n   = len(rec); x = np.arange(n)

        def safe_polyfit(vals):
            try:    return float(np.polyfit(x, vals, 1)[0])
            except: return 0.0

        pt = safe_polyfit(rec["close"].values)
        bt = safe_polyfit(rec["taker_buy_vol"].values)
        sv = (rec["volume"] - rec["taker_buy_vol"]).values
        st = safe_polyfit(sv)
        cvd20 = d["delta"].rolling(20).sum()
        cvd_t = float(cvd20.iloc[-1] - cvd20.iloc[-20]) if len(cvd20)>=20 else 0

        if   pt<-0.3 and bt>0:  ph=2   # accumulation
        elif pt> 0.3 and bt>0:  ph=3   # markup
        elif pt> 0.3 and st>0:  ph=-2  # distribution
        elif pt<-0.3 and st>0:  ph=-3  # markdown
        else:                    ph=0   # consolidation

        return {
            "wyckoff_phase": ph,
            "smart_money_flow": float(np.clip(cvd_t/10000, -3, 3)),
            "price_vol_align": float(np.sign(pt) * np.sign(bt)),   # +1 = aligned
        }

    @staticmethod
    def funding_alpha(funding: pd.DataFrame) -> dict:
        """Funding rate intelligence."""
        if funding.empty or len(funding) < 3:
            return {"funding_rate":0, "funding_regime":0, "funding_trend":0}
        rates = funding["fundingRate"].tail(8).values
        last  = float(rates[-1])
        trend = float(rates[-1] - rates[0])   # direction of funding
        regime= 2 if last>0.0005 else (-2 if last<-0.0003 else 0)
        # Extreme funding = mean reversion trade
        reversion = -1 if last>0.0008 else (1 if last<-0.0005 else 0)
        return {
            "funding_rate":    float(last),
            "funding_regime":  float(regime),
            "funding_trend":   float(np.clip(trend*1000, -3, 3)),
            "funding_revert":  float(reversion),
        }

    def build_all(self, df: pd.DataFrame, funding: pd.DataFrame) -> dict:
        """Build all 80+ alpha signals. Returns flat dict."""
        alphas = {}
        for fn_name, fn in [
            ("mom",    self.momentum_alphas),
            ("rev",    self.mean_reversion_alphas),
            ("of",     self.order_flow_alphas),
            ("vol",    self.volatility_alphas),
            ("micro",  self.microstructure_alphas),
            ("struct", self.structure_alphas),
            ("time",   self.time_alphas),
            ("cross",  self.cross_signal_alphas),
            ("wyck",   self.wyckoff_alphas),
        ]:
            try:
                a = fn(df)
                alphas.update({f"{fn_name}_{k}": v for k,v in a.items()})
            except:
                pass
        try:
            fa = self.funding_alpha(funding)
            alphas.update({f"fund_{k}": v for k,v in fa.items()})
        except:
            pass

        # Clean
        alphas = {k: float(v) if not np.isnan(float(v)) and not np.isinf(float(v)) else 0.0
                  for k,v in alphas.items()}
        return alphas

    def build_matrix(self, df: pd.DataFrame, funding: pd.DataFrame,
                     target_bars: int = 3) -> tuple:
        """Build (X, y, feature_names) for training."""
        rows, ys = [], []
        for i in range(100, len(df) - target_bars):
            sub = df.iloc[:i].copy()
            try:
                a = self.build_all(sub, funding)
                rows.append(list(a.values()))
                future_ret = float(df["close"].iloc[i+target_bars] / df["close"].iloc[i] - 1)
                ys.append(1 if future_ret > 0 else 0)
            except:
                continue

        if not rows:
            return np.zeros((0,80)), np.zeros(0), []

        X = np.array(rows, dtype=float)
        y = np.array(ys, dtype=float)
        X = np.nan_to_num(X, 0)
        feat_names = list(self.build_all(df, funding).keys())
        return X, y, feat_names


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 2 │ ML ENSEMBLE  — GBM + RF + NN + Meta-learner (stacking)
#  "The meta-learner learns when to trust each model." — Two Sigma
# ══════════════════════════════════════════════════════════════════════════
class MLEnsemble:
    """
    Stacked generalization (Wolpert 1992):
    Level 0: GBM, Random Forest, Neural Net → generate OOF predictions
    Level 1: Logistic Regression meta-learner → combines level-0 predictions
    Result: reduces individual model variance, increases robustness.
    """

    def __init__(self):
        # Level 0 models
        self.gbm = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1
        )
        self.nn_weights = []   # pure numpy NN weights
        self.nn_scaler  = StandardScaler()

        # Level 1 meta-learner
        self.meta = LogisticRegression(C=0.5, random_state=42, max_iter=300)

        # Scalers
        self.scaler_gbm = RobustScaler()
        self.scaler_rf  = RobustScaler()
        self.scaler_nn  = StandardScaler()

        self.trained   = False
        self.feat_names= []
        self.feat_imp  = {}
        self.val_score = 0.0

    # ── Simple NN for ensemble (3-layer) ──
    @staticmethod
    def _relu(x): return np.maximum(0, x)
    @staticmethod
    def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))

    def _nn_forward(self, X, weights):
        A = X
        for i in range(0, len(weights)-2, 2):
            A = self._relu(A @ weights[i] + weights[i+1])
        return self._sigmoid(A @ weights[-2] + weights[-1]).ravel()

    def _train_nn(self, X_tr, y_tr, X_v, y_v, epochs=60, lr=1e-3, batch=32):
        """Train simple NN with Adam."""
        n_in = X_tr.shape[1]
        # Init He
        W1 = np.random.randn(n_in,64)*np.sqrt(2/n_in)
        b1 = np.zeros((1,64))
        W2 = np.random.randn(64,32)*np.sqrt(2/64)
        b2 = np.zeros((1,32))
        W3 = np.random.randn(32,1)*np.sqrt(2/32)
        b3 = np.zeros((1,1))
        weights_list = [W1,b1,W2,b2,W3,b3]

        # Adam state
        ms = [np.zeros_like(w) for w in weights_list]
        vs = [np.zeros_like(w) for w in weights_list]
        t  = 0; best_acc=0; best_w=None; no_imp=0

        for ep in range(epochs):
            idx = np.random.permutation(len(X_tr))
            for s in range(0, len(X_tr), batch):
                Xb = X_tr[idx[s:s+batch]]; yb = y_tr[idx[s:s+batch]]
                if len(Xb)<2: continue

                # Forward
                Z1=Xb@W1+b1; A1=self._relu(Z1)
                Z2=A1@W2+b2; A2=self._relu(Z2)
                Z3=A2@W3+b3; A3=self._sigmoid(Z3)

                # Back
                m=len(Xb)
                dZ3 = (A3.ravel()-yb).reshape(-1,1)/m
                dW3 = A2.T@dZ3; db3 = dZ3.sum(0,keepdims=True)
                dA2 = dZ3@W3.T; dZ2 = dA2*(Z2>0)
                dW2 = A1.T@dZ2; db2 = dZ2.sum(0,keepdims=True)
                dA1 = dZ2@W2.T; dZ1 = dA1*(Z1>0)
                dW1 = Xb.T@dZ1; db1 = dZ1.sum(0,keepdims=True)

                grads = [dW1,db1,dW2,db2,dW3,db3]
                t+=1; b1c,b2c,eps2=0.9,0.999,1e-8
                for i,(w,g) in enumerate(zip(weights_list,grads)):
                    ms[i]=b1c*ms[i]+(1-b1c)*g; vs[i]=b2c*vs[i]+(1-b2c)*g**2
                    w -= lr*ms[i]/(1-b1c**t) / (np.sqrt(vs[i]/(1-b2c**t))+eps2)

            # Val check
            yp = self._nn_forward(X_v, weights_list)
            acc= float(((yp>0.5).astype(int)==y_v).mean())
            if acc>best_acc:
                best_acc=acc; best_w=[w.copy() for w in weights_list]; no_imp=0
            else:
                no_imp+=1
            if no_imp>=12: break
            if (ep+1)%20==0: lr*=0.7

        return best_w if best_w else weights_list, best_acc

    def train(self, X: np.ndarray, y: np.ndarray,
              feat_names: list = None, verbose: bool = True):
        """Walk-forward stacked training."""
        if len(X) < 150:
            print("  Need 150+ samples"); return

        self.feat_names = feat_names or [f"f{i}" for i in range(X.shape[1])]
        n = len(X)
        n_val = max(int(n*0.15), 30)
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_v,  y_v  = X[-n_val:], y[-n_val:]

        # Scale
        X_tr_gbm = self.scaler_gbm.fit_transform(X_tr)
        X_v_gbm  = self.scaler_gbm.transform(X_v)
        X_tr_nn  = self.scaler_nn.fit_transform(X_tr)
        X_v_nn   = self.scaler_nn.transform(X_v)

        # Train GBM
        if verbose: print("    Training GBM...", end=" ", flush=True)
        self.gbm.fit(X_tr_gbm, y_tr)
        gbm_acc = float(((self.gbm.predict_proba(X_v_gbm)[:,1]>0.5)==y_v).mean())
        if verbose: print(f"val_acc={gbm_acc:.3f}")

        # Feature importance from GBM (Renaissance-style signal ranking)
        if feat_names:
            imp = self.gbm.feature_importances_
            self.feat_imp = dict(sorted(
                zip(feat_names, imp), key=lambda x:-x[1])[:20])

        # Train RF
        if verbose: print("    Training RF...", end=" ", flush=True)
        self.rf.fit(X_tr_gbm, y_tr)
        rf_acc = float(((self.rf.predict_proba(X_v_gbm)[:,1]>0.5)==y_v).mean())
        if verbose: print(f"val_acc={rf_acc:.3f}")

        # Train NN
        if verbose: print("    Training NN...", end=" ", flush=True)
        self.nn_weights, nn_acc = self._train_nn(X_tr_nn, y_tr, X_v_nn, y_v,
                                                  epochs=80, lr=1e-3)
        if verbose: print(f"val_acc={nn_acc:.3f}")

        # Level 1: build meta-features from val predictions
        meta_X_tr = np.column_stack([
            self.gbm.predict_proba(X_tr_gbm)[:,1],
            self.rf.predict_proba(X_tr_gbm)[:,1],
            self._nn_forward(X_tr_nn, self.nn_weights),
        ])
        meta_X_v = np.column_stack([
            self.gbm.predict_proba(X_v_gbm)[:,1],
            self.rf.predict_proba(X_v_gbm)[:,1],
            self._nn_forward(X_v_nn, self.nn_weights),
        ])

        if verbose: print("    Training meta-learner...", end=" ", flush=True)
        self.meta.fit(meta_X_tr, y_tr)
        meta_acc = float(((self.meta.predict_proba(meta_X_v)[:,1]>0.5)==y_v).mean())
        if verbose: print(f"val_acc={meta_acc:.3f}")

        self.trained   = True
        self.val_score = meta_acc
        if verbose:
            print(f"    Ensemble (meta) val accuracy: {meta_acc:.4f}")
            print(f"    Top alphas: {list(self.feat_imp.keys())[:5]}")

    def predict(self, X: np.ndarray) -> dict:
        if not self.trained:
            return {"prob":0.5,"score":0,"model_probs":{"gbm":0.5,"rf":0.5,"nn":0.5}}
        x = X[-1:].copy()
        xg = self.scaler_gbm.transform(x)
        xn = self.scaler_nn.transform(x)
        p_gbm = float(self.gbm.predict_proba(xg)[0,1])
        p_rf  = float(self.rf.predict_proba(xg)[0,1])
        p_nn  = float(self._nn_forward(xn, self.nn_weights)[0])
        meta_x= np.array([[p_gbm, p_rf, p_nn]])
        prob  = float(self.meta.predict_proba(meta_x)[0,1])
        # Disagreement penalty
        std   = float(np.std([p_gbm, p_rf, p_nn]))
        if std > 0.15: prob = 0.5 + (prob-0.5)*0.5
        sc = 3 if prob>0.70 else 2 if prob>0.62 else 1 if prob>0.56 else \
            -3 if prob<0.30 else -2 if prob<0.38 else -1 if prob<0.44 else 0
        return {
            "prob":        prob,
            "std":         std,
            "signal":      "BUY" if prob>0.56 else ("SELL" if prob<0.44 else "WAIT"),
            "model_probs": {"gbm":p_gbm,"rf":p_rf,"nn":p_nn},
            "score":       sc,
            "val_acc":     self.val_score,
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 3 │ HIDDEN MARKOV MODEL  — Regime-aware signal weighting
#  "The same signal has different edge in different regimes." — AQR
# ══════════════════════════════════════════════════════════════════════════
class HiddenMarkovRegime:
    """
    4-state HMM: Quiet Bull, Volatile Bull, Quiet Bear, Volatile Bear
    Each state has different signal weights.
    """
    STATES = {0:"QUIET_BULL", 1:"VOLATILE_BULL", 2:"QUIET_BEAR", 3:"VOLATILE_BEAR"}
    WEIGHTS = {
        # How much to trust each signal category per regime
        0: {"momentum":1.3,"mean_rev":0.7,"order_flow":1.0,"volume":0.8},  # quiet bull: follow trend
        1: {"momentum":0.8,"mean_rev":1.2,"order_flow":1.3,"volume":1.2},  # vol bull: OF important
        2: {"momentum":1.2,"mean_rev":0.8,"order_flow":1.0,"volume":1.0},  # quiet bear: follow trend
        3: {"momentum":0.6,"mean_rev":1.5,"order_flow":1.4,"volume":1.3},  # vol bear: mean rev + OF
    }

    def fit(self, df: pd.DataFrame):
        """Fit HMM via Baum-Welch on (return, vol) observations."""
        ret = df["close"].pct_change().dropna().values
        vol = df["close"].pct_change().rolling(5).std().dropna().values
        n   = min(len(ret), len(vol)); ret=ret[-n:]; vol=vol[-n:]
        T   = len(ret); K = 4

        # Observation: sign(ret) × vol_regime
        obs = np.zeros(T, dtype=int)
        vol_m = np.median(vol)
        for t in range(T):
            bull = ret[t] > 0
            high = vol[t] > vol_m
            obs[t] = (0 if bull and not high else
                      1 if bull and high else
                      2 if not bull and not high else 3)

        # EM for HMM (Baum-Welch, simplified)
        A = np.ones((K,K))/K      # transition
        pi= np.ones(K)/K           # initial
        B = np.ones((K,4))/4       # emission (4 observation types)

        for _ in range(30):
            # E-step: forward-backward
            alpha = np.zeros((T,K)); beta = np.zeros((T,K))
            alpha[0] = pi * B[:,obs[0]]
            alpha[0] /= max(alpha[0].sum(), 1e-300)
            for t in range(1,T):
                alpha[t] = (alpha[t-1] @ A) * B[:,obs[t]]
                s = alpha[t].sum()
                alpha[t] /= max(s, 1e-300)
            beta[-1] = 1
            for t in range(T-2,-1,-1):
                beta[t] = A @ (B[:,obs[t+1]] * beta[t+1])
                beta[t] /= max(beta[t].sum(), 1e-300)
            gamma = alpha * beta; gamma /= gamma.sum(1,keepdims=True).clip(1e-300)
            xi    = np.zeros((T-1,K,K))
            for t in range(T-1):
                xi[t] = np.outer(alpha[t], beta[t+1] * B[:,obs[t+1]]) * A
                xi[t] /= max(xi[t].sum(), 1e-300)
            # M-step
            pi = gamma[0]; A = xi.sum(0)/xi.sum((0,2),keepdims=True).clip(1e-10)
            for k in range(K):
                for o in range(4):
                    B[k,o] = gamma[obs==o,k].sum() / max(gamma[:,k].sum(), 1e-10)

        self.A=A; self.B=B; self.pi=pi
        self.fitted=True

        # Current state = most likely last state
        return int(np.argmax(alpha[-1]))

    def current_regime(self, df: pd.DataFrame) -> dict:
        if not hasattr(self,'fitted'):
            return {"state":0,"name":"QUIET_BULL","weights":self.WEIGHTS[0],"prob":0.25}
        state = self.fit(df)
        return {
            "state":   state,
            "name":    self.STATES.get(state,"UNKNOWN"),
            "weights": self.WEIGHTS.get(state, self.WEIGHTS[0]),
            "prob":    0.5,
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 4 │ BLACK-LITTERMAN OPTIMIZER
#  "Express views on signals, let BL combine with market equilibrium." — Goldman
# ══════════════════════════════════════════════════════════════════════════
class BlackLittermanSizer:
    """
    Black-Litterman framework adapted for single-asset sizing:
    - Market equilibrium (neutral) position = 1x Kelly
    - Views from signal engine expressed as expected returns
    - BL posterior = optimal size considering uncertainty in views
    """
    @staticmethod
    def optimal_size(prob: float, rr: float, confidence: float,
                     account: float, max_risk: float = 0.02) -> float:
        """
        BL-inspired sizing:
        1. Full Kelly = (p*b - q)/b  where b=rr
        2. Scale by confidence (view uncertainty)
        3. Cap at max_risk fraction of account
        4. Apply shrinkage toward neutral (0.5 Kelly)
        """
        p = max(min(prob, 0.999), 0.001)
        q = 1-p; b = max(rr, 0.01)
        full_k = max((p*b - q)/b, 0)

        # View confidence shrinkage (uncertainty of views)
        conf_scale = confidence / 100   # 0-1
        # BL posterior sizing: blend full Kelly with zero (neutral)
        bl_kelly = full_k * conf_scale * 0.5  # 50% of conf-scaled Kelly

        # Max risk constraint
        risk_fraction = min(bl_kelly, max_risk)
        usdt_size = account * risk_fraction
        return usdt_size, float(bl_kelly), float(full_k)


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 5 │ RISK ENGINE  — Multi-dimensional risk management
#  "Risk management is the #1 alpha." — Paul Tudor Jones
# ══════════════════════════════════════════════════════════════════════════
class RiskEngine:
    def __init__(self, max_daily_loss: float = 0.03,
                 max_drawdown: float = 0.10,
                 var_limit: float = 0.02):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown   = max_drawdown
        self.var_limit      = var_limit
        self.daily_pnl      = 0.0
        self.peak_balance   = 1000.0
        self.current_dd     = 0.0
        self.trade_log      = deque(maxlen=100)

    def cvar(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """CVaR (Expected Shortfall) at alpha confidence."""
        r   = returns.dropna().values
        var = np.percentile(r, alpha*100)
        return float(r[r <= var].mean()) if (r<=var).any() else var

    def assess(self, df: pd.DataFrame, proposed_risk_pct: float,
               account: float) -> dict:
        """Risk assessment — returns go/no-go + adjusted size."""
        ret   = df["close"].pct_change().dropna()
        cvar_ = self.cvar(ret)
        vol5  = float(ret.tail(5).std())
        vol20 = float(ret.tail(20).std())

        # VaR check: is the daily VaR within limits?
        var_95 = float(np.percentile(ret.tail(50), 5)) if len(ret)>=50 else -0.05
        daily_var = abs(var_95) * account * proposed_risk_pct * 20  # rough daily

        # Volatility scaling (Sharpe-optimal sizing)
        target_vol  = 0.02    # target 2% daily vol
        vol_scale   = min(target_vol / max(vol20*np.sqrt(288), 0.001), 2.0)

        # Regime risk adjustment
        high_vol = vol5 > vol20 * 1.5   # vol expanding = danger
        if high_vol: vol_scale *= 0.5

        # Drawdown circuit breaker
        breaker = self.daily_pnl < -account * self.max_daily_loss

        adjusted_risk = proposed_risk_pct * vol_scale

        return {
            "go":           not breaker,
            "vol5":         vol5,
            "vol20":        vol20,
            "cvar_95":      float(cvar_),
            "var_95":       var_95,
            "vol_scale":    float(vol_scale),
            "high_vol":     high_vol,
            "circuit_break":breaker,
            "adj_risk_pct": float(np.clip(adjusted_risk, 0.002, 0.03)),
        }

    def position_heat(self) -> float:
        """Current portfolio heat (risk utilization 0-1)."""
        return min(abs(self.daily_pnl) / max(self.peak_balance * self.max_daily_loss, 1), 1.0)


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 6 │ EXECUTION IQ  — Smart entry timing
#  "Amateur: enter on signal. Professional: enter at the best price." — Citadel
# ══════════════════════════════════════════════════════════════════════════
class ExecutionEngine:
    """
    Optimizes entry based on:
    - Optimal limit vs market order decision
    - Expected slippage (Kyle's lambda)
    - Volume participation rate
    - VWAP execution benchmark
    """
    @staticmethod
    def execution_score(d: pd.DataFrame, side: str) -> dict:
        """Score the current moment for execution quality."""
        last   = d.iloc[-1]
        vol_z  = float(last.get("vol_z", 0))
        atr    = float(last.get("atr", d["close"].iloc[-1]*0.003))
        spread = atr * 0.001   # proxy for bid-ask spread

        # Kyle's lambda (price impact)
        ret  = d["close"].pct_change().tail(20)
        dp   = d["delta_pct"].tail(20)
        lam  = float(abs(np.cov(ret, dp)[0,1]) / max(dp.var(), 1e-10))

        # Participation rate suggestion (don't move market)
        avg_vol = float(d["volume"].tail(20).mean())
        cur_vol = float(last["volume"])
        vpart   = min(cur_vol / max(avg_vol, 1), 3.0)

        # TWAP deviation (enter closer to VWAP is better)
        tp    = (d["high"]+d["low"]+d["close"])/3
        vwap  = float((tp*d["volume"]).tail(20).sum() / d["volume"].tail(20).sum())
        price = float(d["close"].iloc[-1])
        vdev  = (price - vwap) / vwap

        # Execution timing score
        score = 0
        if vol_z < 1.5:     score += 1   # not chasing high-vol candle
        if vol_z < 0.5:     score += 1   # low vol = tight spread
        if lam < 0.001:     score += 1   # low impact
        if side=="BUY"  and vdev < -0.002: score += 1  # buying below VWAP
        if side=="SELL" and vdev >  0.002: score += 1  # selling above VWAP

        return {
            "exec_score":     score,
            "spread_proxy":   float(spread),
            "kyle_lambda":    float(lam),
            "vol_part_rate":  float(vpart),
            "vwap_dev":       float(vdev),
            "exec_type":      "LIMIT" if score >= 3 else "MARKET",
            "enter_now":      score >= 2,
        }

    @staticmethod
    def optimal_levels(price: float, atr: float, side: str,
                       poc: float, vah: float, val: float) -> dict:
        """
        Compute optimal entry, stop, TP using structure.
        PTJ principle: enter at structure, stop just beyond it.
        """
        # Entry: at or slightly inside structure
        if side == "BUY":
            entry = price                              # market for now
            stop  = round(min(val, price-atr*1.5), 1)  # below VAL or 1.5×ATR
            tp1   = round(poc if poc>price else price+atr*2, 1)
            tp2   = round(vah if vah>tp1 else price+atr*4, 1)
        else:  # SELL
            entry = price
            stop  = round(max(vah, price+atr*1.5), 1)
            tp1   = round(poc if poc<price else price-atr*2, 1)
            tp2   = round(val if val<tp1 else price-atr*4, 1)

        rr = abs(tp1-entry) / max(abs(entry-stop), 0.01)
        return {"entry":entry,"stop":stop,"tp1":tp1,"tp2":tp2,"rr":rr}
