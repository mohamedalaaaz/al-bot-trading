"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   BTC/USDT APEX TRADING ENGINE — INSTITUTIONAL + AI FUSION                     ║
║   Synthesized from 5 trading systems into one unified architecture              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  LAYER 1 │ DATA ENGINE          — Multi-TF Binance Futures fetch + WebSocket   ║
║  LAYER 2 │ FEATURE FORGE        — 40+ engineered signals (CVD, delta, ATR...)  ║
║  LAYER 3 │ ORDER FLOW SUITE     — Absorption, imbalance, big traders, iceberg  ║
║  LAYER 4 │ MARKET STRUCTURE     — VWAP/TWAP, Market Profile, TPO, Value Area   ║
║  LAYER 5 │ LIQUIDITY MAP        — Stop clusters, equal H/L, round magnets      ║
║  LAYER 6 │ PATTERN MINER        — Time/day bias, liquidation, CVD divergence   ║
║  LAYER 7 │ AI ENSEMBLE          — RandomForest + LSTM hybrid confidence score  ║
║  LAYER 8 │ WYCKOFF / HEDGE FUND — Accumulation, distribution, smart money      ║
║  LAYER 9 │ SIGNAL ENGINE        — Multi-factor composite score (-10 to +10)    ║
║  LAYER 10│ RISK MANAGER         — ATR-based position sizing, max drawdown      ║
║  LAYER 11│ EXECUTION ENGINE     — Order placement + stop management            ║
║  LAYER 12│ ALERT SYSTEM         — Telegram notifications + live logging        ║
╚══════════════════════════════════════════════════════════════════════════════════╝

INSTALLATION:
    pip install requests pandas numpy scikit-learn tensorflow ta joblib websocket-client python-binance

USAGE:
    python btc_apex_bot.py --mode live      # Real trading (needs API keys)
    python btc_apex_bot.py --mode paper     # Paper trading with live data (DEFAULT)
    python btc_apex_bot.py --mode backtest  # Backtest on historical data
    python btc_apex_bot.py --mode analyze   # One-shot analysis only
"""

# ═══════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════
import os, sys, argparse, time, json, hmac, hashlib, threading, logging
import warnings; warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import websocket

# ── ML / AI ──────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠  scikit-learn not found. AI layer disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as keras_load
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# ── Technical indicators (optional but recommended) ───────────────
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Exchange
    symbol:         str   = "BTCUSDT"
    base_url:       str   = "https://fapi.binance.com"
    api_key:        str   = ""          # set via env var BINANCE_API_KEY
    api_secret:     str   = ""          # set via env var BINANCE_SECRET
    ws_url:         str   = "wss://fstream.binance.com/ws"

    # Trading mode
    mode:           str   = "paper"     # "live" | "paper" | "backtest" | "analyze"
    symbols:        list  = field(default_factory=lambda: ["BTCUSDT"])

    # Risk parameters
    balance:        float = 10_000.0
    risk_pct:       float = 0.01        # 1% per trade
    max_open_trades:int   = 3
    max_daily_loss: float = 0.05        # 5% daily stop-out
    atr_stop_mult:  float = 1.5
    atr_tp1_mult:   float = 2.0
    atr_tp2_mult:   float = 4.0
    min_rr:         float = 1.5         # minimum reward:risk ratio to take trade

    # Signal thresholds
    min_score:      int   = 3           # abs score required to trade
    min_ai_conf:    float = 0.60        # AI probability threshold
    imbalance_thr:  float = 3.0
    big_trade_z:    float = 5.0

    # Data lookback
    lookback_1m:    int   = 1500
    lookback_5m:    int   = 1000
    lookback_1h:    int   = 500

    # AI / ML
    lstm_lookback:  int   = 50          # candles fed to LSTM
    retrain_every:  int   = 200         # candles between retrains
    model_dir:      str   = "./models"

    # Telegram alerts
    telegram:       bool  = False
    tg_token:       str   = ""
    tg_chat:        str   = ""

    # Logging
    log_level:      str   = "INFO"
    log_file:       str   = "apex_bot.log"

CFG = Config(
    api_key    = os.environ.get("BINANCE_API_KEY", ""),
    api_secret = os.environ.get("BINANCE_SECRET", ""),
    tg_token   = os.environ.get("TG_TOKEN", ""),
    tg_chat    = os.environ.get("TG_CHAT", ""),
    telegram   = bool(os.environ.get("TG_TOKEN")),
)

# ═══════════════════════════════════════════════════════════════════
#  LOGGING  (UTF-8 safe on Windows CP1252 terminals)
# ═══════════════════════════════════════════════════════════════════
import io as _io
_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

# Console handler — wrap stdout in UTF-8 so box-drawing chars work on Windows
_con = logging.StreamHandler(
    _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_con.setFormatter(_fmt)

# File handler — always UTF-8
_fh = logging.FileHandler(CFG.log_file, encoding="utf-8")
_fh.setFormatter(_fmt)

_root = logging.getLogger()
_root.setLevel(getattr(logging, CFG.log_level))
_root.handlers.clear()
_root.addHandler(_con)
_root.addHandler(_fh)
log = logging.getLogger("APEX")

# ═══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════
@dataclass
class TradePosition:
    symbol:    str
    side:      str          # "BUY" | "SELL"
    entry:     float
    stop:      float
    tp1:       float
    tp2:       float
    qty:       float
    score:     int
    ai_conf:   float
    opened_at: datetime     = field(default_factory=lambda: datetime.now(timezone.utc))
    tp1_hit:   bool         = False
    pnl:       float        = 0.0
    status:    str          = "OPEN"    # "OPEN" | "CLOSED" | "STOPPED"

@dataclass
class SignalResult:
    bias:       str         # "STRONG LONG" | "LONG" | "NEUTRAL" | "SHORT" | "STRONG SHORT"
    score:      int         # -10 to +10
    confidence: float       # 0 to 1
    ai_prob:    float       # 0 to 1 (LSTM + RF ensemble)
    reasons:    List[str]   = field(default_factory=list)
    entry:      Optional[float] = None
    stop:       Optional[float] = None
    tp1:        Optional[float] = None
    tp2:        Optional[float] = None
    rr:         float       = 0.0

# ═══════════════════════════════════════════════════════════════════
#  LAYER 1 │ DATA ENGINE
# ═══════════════════════════════════════════════════════════════════
class DataEngine:
    """Fetches and manages multi-timeframe OHLCV data from Binance Futures."""

    INTERVAL_MINS = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ApexBot/2.0"})

    def fetch_klines(self, symbol: str, interval: str, limit: int = 1500,
                     start_ms: Optional[int] = None) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_ms:
            params["startTime"] = int(start_ms)
        r = self.session.get(f"{self.cfg.base_url}/fapi/v1/klines",
                             params=params, timeout=12)
        r.raise_for_status()
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_vol","tbqv","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume","taker_buy_vol","trades"]:
            df[c] = df[c].astype(float)
        return df.drop(columns=["close_time","quote_vol","tbqv","ignore"])

    def fetch_multi_tf(self, symbol: str, target: int, interval: str) -> pd.DataFrame:
        mins  = self.INTERVAL_MINS.get(interval, 60)
        start = datetime.now(timezone.utc) - timedelta(minutes=target * mins + 60)
        dfs, cur_ms = [], start.timestamp() * 1000
        for _ in range(10):
            try:
                df = self.fetch_klines(symbol, interval, 1500, cur_ms)
                if df.empty: break
                dfs.append(df)
                cur_ms = df["open_time"].iloc[-1].timestamp() * 1000 + mins * 60_000
                if len(df) < 1500: break
            except Exception as e:
                log.warning(f"fetch batch error: {e}")
                break
        if not dfs:
            return pd.DataFrame()
        out = pd.concat(dfs).drop_duplicates("open_time").sort_values("open_time")
        return out.tail(target).reset_index(drop=True)

    def fetch_funding(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        r = self.session.get(f"{self.cfg.base_url}/fapi/v1/fundingRate",
                             params={"symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        return df

    def fetch_open_interest(self, symbol: str) -> dict:
        r = self.session.get(f"{self.cfg.base_url}/fapi/v1/openInterest",
                             params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        return r.json()

    def load_all(self, symbol: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, bool]:
        log.info("▶ Connecting to Binance Futures…")
        live = False
        dfs  = {}
        configs = [("1m", self.cfg.lookback_1m), ("5m", self.cfg.lookback_5m),
                   ("1h", self.cfg.lookback_1h), ("4h", 200)]
        for tf, lim in configs:
            try:
                df = self.fetch_multi_tf(symbol, lim, tf)
                if not df.empty:
                    dfs[tf] = df
                    live    = True
                    log.info(f"  {tf:>3} ✓  {len(df):>5} candles  "
                             f"{df['open_time'].min().date()} → {df['open_time'].max().date()}")
            except Exception as e:
                log.warning(f"  {tf} fetch failed: {e}")

        funding = pd.DataFrame()
        if live:
            try:
                funding = self.fetch_funding(symbol)
                log.info(f"  Funding ✓  {len(funding)} records")
            except Exception as e:
                log.warning(f"  Funding fetch failed: {e}")

        if not live:
            log.warning("  No network — using synthetic data (run locally for live)")
            dfs, funding = self._synthetic_data()

        return dfs, funding, live

    def _synthetic_data(self):
        """High-quality synthetic BTC data for offline testing."""
        np.random.seed(42)
        def synth(n, mins, base=67000, seed=0):
            np.random.seed(seed)
            dates = pd.date_range(
                end=datetime.now(timezone.utc).replace(second=0, microsecond=0),
                periods=n, freq=f"{mins}min", tz="UTC"
            )
            price = float(base)
            rows  = []
            for dt in dates:
                h  = dt.hour
                sv = 2.2 if h in [8,9,13,14,15,16] else (0.5 if h in [1,2,3,4] else 1.0)
                mu = -0.00015 if h in [16,17,18] else 0.00008
                price = max(price * (1 + np.random.normal(mu, 0.0028*sv)), 50000)
                hi = price*(1+abs(np.random.normal(0,0.0022*sv)))
                lo = price*(1-abs(np.random.normal(0,0.0022*sv)))
                op = price*(1+np.random.normal(0,0.0008))
                vol = max(abs(np.random.normal(1200,450))*sv, 80)
                bsk = 0.65 if h in [8,9,10] else (0.36 if h in [17,18,19] else 0.50)
                tb  = vol * np.clip(np.random.beta(bsk*7,(1-bsk)*7), 0.05, 0.95)
                if np.random.random() < 0.03:
                    vol *= np.random.uniform(4, 9); tb *= np.random.uniform(3, 8)
                rows.append({"open_time":dt,"open":op,"high":hi,"low":lo,"close":price,
                             "volume":vol,"taker_buy_vol":tb,"trades":int(vol/0.05)})
            return pd.DataFrame(rows)

        dfs = {"1m": synth(1500,1,67200,1), "5m": synth(1000,5,67000,2),
               "1h": synth(500,60,66800,3), "4h": synth(200,240,65000,4)}

        # Synthetic funding
        times, rates, prev = [], [], 0.0001
        for dt in dfs["1h"]["open_time"]:
            if dt.hour in [0, 8, 16]:
                prev = np.clip(prev*0.7 + np.random.normal(0.0001,0.00025), -0.002, 0.002)
                times.append(dt); rates.append(prev)
        funding = pd.DataFrame({"fundingTime": times, "fundingRate": rates})
        return dfs, funding


# ═══════════════════════════════════════════════════════════════════
#  LAYER 2 │ FEATURE FORGE
# ═══════════════════════════════════════════════════════════════════
class FeatureForge:
    """Computes all raw features from OHLCV data."""

    @staticmethod
    def build_base(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Candle anatomy
        d["body"]       = d["close"] - d["open"]
        d["body_pct"]   = d["body"] / d["open"] * 100
        d["is_bull"]    = d["body"] > 0
        d["range"]      = d["high"] - d["low"]
        d["range_pct"]  = d["range"] / d["open"] * 100
        d["wick_top"]   = d["high"] - d[["open","close"]].max(axis=1)
        d["wick_bot"]   = d[["open","close"]].min(axis=1) - d["low"]
        d["wick_ratio"] = d["wick_top"] / (d["wick_bot"].replace(0, 1e-9))

        # Volume delta
        d["sell_vol"]   = d["volume"] - d["taker_buy_vol"]
        d["delta"]      = d["taker_buy_vol"] - d["sell_vol"]
        d["delta_pct"]  = d["delta"] / d["volume"].replace(0, np.nan)

        # ATR (Wilder's smoothing)
        hl  = d["high"] - d["low"]
        hpc = (d["high"] - d["close"].shift(1)).abs()
        lpc = (d["low"]  - d["close"].shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        d["atr"]        = tr.ewm(span=14, adjust=False).mean()
        d["atr_pct"]    = d["atr"] / d["close"] * 100

        # Volume stats
        d["vol_sma50"]  = d["volume"].rolling(50).mean()
        d["vol_std50"]  = d["volume"].rolling(50).std()
        d["vol_z"]      = (d["volume"] - d["vol_sma50"]) / d["vol_std50"].replace(0, 1e-9)
        d["vol_ratio"]  = d["volume"] / d["vol_sma50"].replace(0, np.nan)

        # Momentum
        for p in [3, 5, 10, 20]:
            d[f"ret_{p}"] = d["close"].pct_change(p) * 100

        # Session
        d["hour"]    = d["open_time"].dt.hour
        d["dow"]     = d["open_time"].dt.day_name()
        d["session"] = d["hour"].map(lambda h:
            "Asia" if 0 <= h < 8 else "London" if 8 <= h < 13 else
            "NY" if 13 <= h < 20 else "Late")

        # ── Technical indicators (if ta available) ──────────────
        if TA_AVAILABLE:
            d["rsi"]     = ta.momentum.RSIIndicator(d["close"], 14).rsi()
            d["ema_20"]  = ta.trend.EMAIndicator(d["close"], 20).ema_indicator()
            d["ema_50"]  = ta.trend.EMAIndicator(d["close"], 50).ema_indicator()
            d["ema_200"] = ta.trend.EMAIndicator(d["close"], 200).ema_indicator()
            macd = ta.trend.MACD(d["close"])
            d["macd"]    = macd.macd()
            d["macd_sig"]= macd.macd_signal()
            d["macd_hist"]= macd.macd_diff()
            bb = ta.volatility.BollingerBands(d["close"], 20)
            d["bb_upper"]= bb.bollinger_hband()
            d["bb_lower"]= bb.bollinger_lband()
            d["bb_width"]= (d["bb_upper"] - d["bb_lower"]) / d["close"] * 100
            d["bb_pct"]  = bb.bollinger_pband()
            d["adx"]     = ta.trend.ADXIndicator(d["high"],d["low"],d["close"],14).adx()
            d["stoch_k"] = ta.momentum.StochasticOscillator(
                d["high"],d["low"],d["close"],14).stoch()
        else:
            # Fallback manual RSI
            delta = d["close"].diff()
            gain  = delta.where(delta > 0, 0).ewm(span=14).mean()
            loss  = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
            rs    = gain / loss.replace(0, 1e-9)
            d["rsi"] = 100 - (100 / (1 + rs))

            d["ema_20"]  = d["close"].ewm(span=20, adjust=False).mean()
            d["ema_50"]  = d["close"].ewm(span=50, adjust=False).mean()
            d["ema_200"] = d["close"].ewm(span=200, adjust=False).mean()

            # MACD
            ema12 = d["close"].ewm(span=12, adjust=False).mean()
            ema26 = d["close"].ewm(span=26, adjust=False).mean()
            d["macd"]     = ema12 - ema26
            d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
            d["macd_hist"]= d["macd"] - d["macd_sig"]

            # Bollinger Bands
            sma20 = d["close"].rolling(20).mean()
            std20 = d["close"].rolling(20).std()
            d["bb_upper"]= sma20 + 2 * std20
            d["bb_lower"]= sma20 - 2 * std20
            d["bb_width"]= (d["bb_upper"] - d["bb_lower"]) / d["close"] * 100
            d["bb_pct"]  = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

        return d.dropna().reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
#  LAYER 3 │ ORDER FLOW SUITE
# ═══════════════════════════════════════════════════════════════════
class OrderFlowSuite:
    """CVD Pro + Big Traders + Absorption + Imbalance detection."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = self._cvd_pro(d)
        d = self._big_traders(d)
        d = self._absorption_engine(d)
        d = self._imbalance_engine(d)
        d = self._liquidity_sweeps(d)
        d = self._trapped_traders(d)
        return d

    def _cvd_pro(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["cvd"]          = d["delta"].cumsum()
        d["cvd_roll20"]   = d["delta"].rolling(20).sum()
        d["cvd_roll5"]    = d["delta"].rolling(5).sum()
        d["cvd_slope3"]   = d["cvd_roll20"].diff(3)
        d["price_slope3"] = d["close"].diff(3) / d["close"].shift(3) * 100

        # Divergences
        d["div_bull"]     = (d["price_slope3"] < -0.15) & (d["cvd_slope3"] > 0)
        d["div_bear"]     = (d["price_slope3"] >  0.15) & (d["cvd_slope3"] < 0)

        # VWAP/delta below/above divergence
        if "vwap20_dev" in d.columns:
            d["div_bull_vwap"]= (~d["is_bull"]) & (d["delta_pct"] > 0.1) & (d["vwap20_dev"] < -0.2)
            d["div_bear_vwap"]= d["is_bull"] & (d["delta_pct"] < -0.1) & (d["vwap20_dev"] > 0.2)
        else:
            d["div_bull_vwap"] = False
            d["div_bear_vwap"] = False

        # Exhaustion: candle direction vs delta mismatch
        d["delta_exhaust"]= (
            ((d["is_bull"])  & (d["delta_pct"] < 0.05)) |
            ((~d["is_bull"]) & (d["delta_pct"] > -0.05))
        )

        # Divergence composite score
        d["div_score_bull"]= d["div_bull"].astype(int) + d["div_bull_vwap"].astype(int)
        d["div_score_bear"]= d["div_bear"].astype(int) + d["div_bear_vwap"].astype(int)
        return d

    def _big_traders(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        z  = self.cfg.big_trade_z
        d["is_whale"]     = d["vol_z"] > z
        d["is_large"]     = (d["vol_z"] > 2.5) & (d["vol_z"] <= z)

        # Iceberg: high volume + many trades + small avg trade size
        avg_trade   = d["volume"] / d["trades"].replace(0, np.nan)
        global_avg  = avg_trade.median()
        d["avg_trade_sz"] = avg_trade
        d["iceberg"]      = (
            (d["vol_z"] > 1.5) &
            (d["trades"] > d["trades"].rolling(50).mean() * 1.5) &
            (avg_trade < global_avg * 0.7)
        )
        # Spoof: price direction vs delta completely opposite
        d["spoof_suspect"]= (
            (d["is_bull"] & (d["delta_pct"] < -0.3)) |
            (~d["is_bull"] & (d["delta_pct"] > 0.3))
        )
        # Accumulation vs distribution
        d["accum_signal"] = (~d["is_bull"]) & (d["delta_pct"] > 0.15) & (d["vol_z"] > 0.5)
        d["distrib_signal"]= d["is_bull"] & (d["delta_pct"] < -0.15) & (d["vol_z"] > 0.5)
        return d

    def _absorption_engine(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Absorption: huge vol + tiny body = price absorbed the order flow
        d["absorption"]  = (d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.1)

        # Bid absorption: falls to low, closes near high, buyers absorbed sellers
        d["bid_absorb"]  = (
            (d["low"] < d["open"] * 0.999) &
            (d["close"] > d["open"] * 0.999) &
            (d["delta_pct"] > 0.1) & (d["vol_z"] > 1.0)
        )
        # Ask absorption: rallies to high, closes near low, sellers absorbed buyers
        d["ask_absorb"]  = (
            (d["high"] > d["open"] * 1.001) &
            (d["close"] < d["open"] * 1.001) &
            (d["delta_pct"] < -0.1) & (d["vol_z"] > 1.0)
        )
        # Exhaustion
        d["buy_exhaust"] = (d["delta_pct"] > 0.3) & (d["body_pct"] < 0.05)
        d["sell_exhaust"]= (d["delta_pct"] < -0.3) & (d["body_pct"] > -0.05)
        return d

    def _imbalance_engine(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        thr = self.cfg.imbalance_thr
        # Stacked imbalances: 3 consecutive candles all same direction delta
        d["delta_pos"]    = d["delta_pct"] > 0.1
        d["stacked_buy"]  = d["delta_pos"].rolling(3).sum() == 3
        d["stacked_sell"] = (~d["delta_pos"]).rolling(3).sum() == 3
        # Strong stacked: 5 consecutive
        d["super_stack_buy"] = d["delta_pos"].rolling(5).sum() == 5
        d["super_stack_sell"]= (~d["delta_pos"]).rolling(5).sum() == 5
        return d

    def _liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Sweep = price exceeded prior high/low but closed back inside
        d["sweep_high"]  = (
            (d["high"] > d["high"].rolling(10).max().shift(1)) &
            (d["close"] < d["high"].shift(1))
        )
        d["sweep_low"]   = (
            (d["low"] < d["low"].rolling(10).min().shift(1)) &
            (d["close"] > d["low"].shift(1))
        )
        d["sweep_high5"] = (
            (d["high"] > d["high"].rolling(5).max().shift(1)) &
            (d["close"] < d["high"].shift(1))
        )
        d["sweep_low5"]  = (
            (d["low"] < d["low"].rolling(5).min().shift(1)) &
            (d["close"] > d["low"].shift(1))
        )
        return d

    def _trapped_traders(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Trapped longs: strong up candle then full reversal
        d["trapped_longs"] = (
            (d["body_pct"].shift(1) > 0.25) &
            (d["close"] < d["open"].shift(1))
        )
        d["trapped_shorts"]= (
            (d["body_pct"].shift(1) < -0.25) &
            (d["close"] > d["open"].shift(1))
        )
        return d


# ═══════════════════════════════════════════════════════════════════
#  LAYER 4 │ MARKET STRUCTURE
# ═══════════════════════════════════════════════════════════════════
class MarketStructure:
    """VWAP/TWAP, Market Profile, TPO, Value Area computation."""

    def compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        d   = df.copy()
        tp  = (d["high"] + d["low"] + d["close"]) / 3

        for n in [20, 50, 200]:
            if len(d) >= n:
                d[f"vwap{n}"]     = (tp * d["volume"]).rolling(n).sum() \
                                   / d["volume"].rolling(n).sum()
                d[f"vwap{n}_dev"] = (d["close"] - d[f"vwap{n}"]) / d[f"vwap{n}"] * 100

        # Session VWAP
        d["date"]        = d["open_time"].dt.date
        d["sess_tp_vol"] = tp * d["volume"]
        d["sess_vwap"]   = d.groupby("date")["sess_tp_vol"].cumsum() \
                         / d.groupby("date")["volume"].cumsum()
        d["sess_vwap_dev"]= (d["close"] - d["sess_vwap"]) / d["sess_vwap"] * 100

        # TWAP
        d["twap20"]      = tp.rolling(20).mean()

        # VWAP bands (volume-weighted std dev)
        vwap_vol = (d["volume"] * (tp - d.get("vwap20", tp))**2).rolling(20).sum()
        vol_sum  = d["volume"].rolling(20).sum()
        d["vwap_std"]    = np.sqrt(vwap_vol / vol_sum.replace(0, np.nan))
        d["vwap20"]      = d.get("vwap20", tp.rolling(20).mean())
        d["vwap_u1"]     = d["vwap20"] + d["vwap_std"]
        d["vwap_l1"]     = d["vwap20"] - d["vwap_std"]
        d["vwap_u2"]     = d["vwap20"] + 2 * d["vwap_std"]
        d["vwap_l2"]     = d["vwap20"] - 2 * d["vwap_std"]

        # Position relative to all VWAPs
        if "vwap20" in d.columns:
            d["above_vwap20"]  = d["close"] > d["vwap20"]
        if "vwap50" in d.columns:
            d["above_vwap50"]  = d["close"] > d["vwap50"]
        if "vwap200" in d.columns:
            d["above_vwap200"] = d["close"] > d["vwap200"]

        return d

    def compute_market_profile(self, df: pd.DataFrame,
                               tick_size: float = 25.0) -> Tuple[float, float, float]:
        """Returns (POC, VAH, VAL)."""
        lo  = df["low"].min()
        hi  = df["high"].max()
        buckets = np.arange(np.floor(lo/tick_size)*tick_size,
                            np.ceil(hi/tick_size)*tick_size + tick_size, tick_size)

        vol_profile = defaultdict(float)
        for _, row in df.iterrows():
            in_range = buckets[(buckets >= row["low"]) & (buckets <= row["high"])]
            if not len(in_range): continue
            per = row["volume"] / len(in_range)
            for b in in_range:
                vol_profile[b] += per

        if not vol_profile:
            mid = (lo + hi) / 2
            return mid, hi, lo

        profile_df  = pd.DataFrame({"price": list(vol_profile.keys()),
                                    "volume": list(vol_profile.values())}).sort_values("price")
        total_vol   = profile_df["volume"].sum()
        poc         = profile_df.loc[profile_df["volume"].idxmax(), "price"]

        # Value Area (70%)
        poc_idx     = profile_df.index[profile_df["price"] == poc][0]
        va_rows     = {poc_idx}
        cum_vol     = profile_df.loc[poc_idx, "volume"]
        while cum_vol / total_vol < 0.70:
            ul = max(va_rows) + 1
            ll = min(va_rows) - 1
            up = profile_df.loc[ul, "volume"] if ul in profile_df.index else 0
            dn = profile_df.loc[ll, "volume"] if ll in profile_df.index else 0
            if up >= dn and ul in profile_df.index:
                va_rows.add(ul); cum_vol += up
            elif ll in profile_df.index:
                va_rows.add(ll); cum_vol += dn
            else:
                break

        va_df = profile_df.loc[list(va_rows)]
        return poc, va_df["price"].max(), va_df["price"].min()

    def compute_liquidity_map(self, df: pd.DataFrame) -> Tuple[List, List, List]:
        """Returns (swing_highs_near, swing_lows_near, round_levels)."""
        price     = df["close"].iloc[-1]
        atr       = df["atr"].iloc[-1]
        tolerance = atr * 0.3

        eq_highs, eq_lows = [], []
        for i in range(5, len(df)-2):
            h = df["high"].iloc[i]
            if ((df["high"].iloc[max(0,i-10):i] - h).abs() < tolerance).any():
                eq_highs.append(round(h, -2))
            l = df["low"].iloc[i]
            if ((df["low"].iloc[max(0,i-10):i] - l).abs() < tolerance).any():
                eq_lows.append(round(l, -2))

        lookback = min(10, len(df)//4)
        swing_h, swing_l = [], []
        for i in range(lookback, len(df)-lookback):
            if df["high"].iloc[i] == df["high"].iloc[i-lookback:i+lookback].max():
                swing_h.append(df["high"].iloc[i])
            if df["low"].iloc[i] == df["low"].iloc[i-lookback:i+lookback].min():
                swing_l.append(df["low"].iloc[i])

        near_sh = sorted(swing_h, key=lambda x: abs(x-price))[:5]
        near_sl = sorted(swing_l, key=lambda x: abs(x-price))[:5]

        round_lvls = []
        for mult in [100, 250, 500, 1000]:
            lo_r = np.floor(price/mult) * mult
            hi_r = np.ceil(price/mult)  * mult
            round_lvls.extend([lo_r-mult, lo_r, hi_r, hi_r+mult])
        round_lvls = sorted(set(r for r in round_lvls if abs(r-price) < atr*15))

        return near_sh, near_sl, round_lvls


# ═══════════════════════════════════════════════════════════════════
#  LAYER 5 │ PATTERN MINER
# ═══════════════════════════════════════════════════════════════════
class PatternMiner:
    """Statistical edge discovery: time-of-day, DOW, liquidation, streaks."""

    def compute_edges(self, df: pd.DataFrame) -> dict:
        d = df.copy()
        d["next_bull"]   = d["close"].shift(-1) > d["close"]
        d["next_return"] = d["close"].shift(-1) / d["close"] - 1

        edges = {}

        # Time-of-day edge
        tod = d.groupby("hour").agg(
            bull_rate=("next_bull","mean"),
            avg_ret  =("next_return","mean")
        ).reset_index()
        tod["edge"] = (tod["bull_rate"] - 0.5).abs()
        edges["tod"] = tod.sort_values("edge", ascending=False)

        # Day-of-week
        dow = d.groupby("dow").agg(bull_rate=("next_bull","mean")).reset_index()
        edges["dow"] = dow.sort_values("bull_rate", ascending=False)

        # Streak mean-reversion
        streaks, cur, cnt = [], None, 0
        for _, row in d.iterrows():
            dir_ = "bull" if row["is_bull"] else "bear"
            cnt  = cnt+1 if dir_ == cur else 1
            cur  = dir_
            streaks.append({"dir": cur, "len": min(cnt, 5)})
        d["streak_dir"] = [s["dir"] for s in streaks]
        d["streak_len"] = [s["len"] for s in streaks]
        revert_rows = []
        for dir_ in ["bull","bear"]:
            for n in range(1,6):
                sub = d[(d["streak_dir"]==dir_) & (d["streak_len"]==n)]
                if len(sub) >= 5:
                    mr = (1-sub["next_bull"].mean()) if dir_=="bull" else sub["next_bull"].mean()
                    revert_rows.append({"after":f"{n}x {dir_}","count":len(sub),"revert_prob":mr})
        edges["streaks"] = pd.DataFrame(revert_rows).sort_values("revert_prob", ascending=False) \
                          if revert_rows else pd.DataFrame()

        # Liquidation zone: what happens after vol spike
        for thresh in [1.5, 2.5, 3.5]:
            sub = d[d["vol_z"] > thresh]
            if len(sub) >= 3:
                edges[f"liq_{thresh:.0f}"] = {
                    "count":     len(sub),
                    "bull_rate": sub["next_bull"].mean(),
                    "avg_ret":   sub["next_return"].mean() * 100
                }

        return edges

    def current_session_edge(self, df: pd.DataFrame, edges: dict) -> float:
        """Return edge score for current hour/dow combo."""
        if df.empty or "tod" not in edges:
            return 0.0
        hour = df["hour"].iloc[-1]
        tod  = edges["tod"]
        row  = tod[tod["hour"] == hour]
        if row.empty:
            return 0.0
        bull_rate = row["bull_rate"].values[0]
        return (bull_rate - 0.5) * 2  # -1 to +1


# ═══════════════════════════════════════════════════════════════════
#  LAYER 6 │ WYCKOFF / HEDGE FUND ANALYSIS
# ═══════════════════════════════════════════════════════════════════
class WyckoffAnalysis:
    """Institutional behavior detection using Wyckoff methodology."""

    def analyze(self, df: pd.DataFrame, poc: float, vah: float, val: float,
                funding: pd.DataFrame) -> Tuple[int, int, bool]:
        """Returns (wyckoff_bias, funding_bias, cvd_net_bull)."""
        d = df.copy()
        recent = d.tail(30)
        if len(recent) < 10:
            return 0, 0, False

        n    = len(recent)
        x    = np.arange(n)
        price_trend   = np.polyfit(x, recent["close"].values, 1)[0]
        buy_vol_trend = np.polyfit(x, recent["taker_buy_vol"].values, 1)[0]
        sell_vol_trend= np.polyfit(x, recent["sell_vol"].values, 1)[0]

        if   price_trend < -0.5 and buy_vol_trend > 0:  wyck = 1   # Accumulation
        elif price_trend > 0.5  and sell_vol_trend > 0: wyck = -1  # Distribution
        elif price_trend < 0    and sell_vol_trend < 0: wyck = -2  # Markdown
        elif price_trend > 0    and buy_vol_trend > 0:  wyck = 2   # Markup
        else:                                            wyck = 0   # Consolidation

        # Funding rate
        fund_bias = 0
        if not funding.empty:
            avg_fr = funding["fundingRate"].tail(8).mean()
            if avg_fr > 0.0005:   fund_bias = -1   # overheated longs → bear revert
            elif avg_fr < -0.0003: fund_bias = 1   # squeezed shorts → bull revert

        # CVD net
        d["cvd_20"]  = d["delta"].rolling(20).sum()
        cvd_now  = d["cvd_20"].iloc[-1]
        cvd_prev = d["cvd_20"].iloc[-20] if len(d) > 20 else cvd_now
        cvd_bull = cvd_now > cvd_prev

        return wyck, fund_bias, cvd_bull


# ═══════════════════════════════════════════════════════════════════
#  LAYER 7 │ AI ENSEMBLE
# ═══════════════════════════════════════════════════════════════════
class AIEnsemble:
    """
    Hybrid ML system:
      - RandomForest on 20+ engineered features (fast, interpretable)
      - LSTM on sequence of last N candles (pattern memory)
    Final probability = weighted average of both models.
    """

    FEATURE_COLS = [
        "delta_pct","vol_z","body_pct","wick_ratio","rsi",
        "macd_hist","bb_pct","ret_3","ret_5","ret_10",
        "cvd_slope3","price_slope3","vol_ratio","atr_pct",
        "sess_vwap_dev" if "sess_vwap_dev" else "body_pct",
    ]

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.rf_model  = None
        self.rf_pipe   = None
        self.lstm_model= None
        self.scaler    = MinMaxScaler() if ML_AVAILABLE else None
        self.candle_count = 0
        os.makedirs(cfg.model_dir, exist_ok=True)

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            available = ["delta_pct","vol_z","body_pct","rsi"]
            available = [c for c in available if c in df.columns]
        return df[available].fillna(0)

    def _make_target(self, df: pd.DataFrame, horizon: int = 3) -> pd.Series:
        return (df["close"].shift(-horizon) > df["close"]).astype(int)

    def train(self, df: pd.DataFrame):
        if not ML_AVAILABLE or len(df) < 100:
            return

        log.info("  Training AI ensemble…")
        X = self._get_features(df)
        y = self._make_target(df)
        valid = X.index.intersection(y.dropna().index)
        X, y = X.loc[valid], y.loc[valid]
        if len(X) < 60:
            return

        X_train = X.iloc[:-3]
        y_train = y.iloc[:-3]

        # ── RandomForest ─────────────────────────────────────────
        try:
            self.rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            rf_path = os.path.join(self.cfg.model_dir, "rf_model.joblib")
            joblib.dump(self.rf_model, rf_path)
        except Exception as e:
            log.warning(f"RF training failed: {e}")

        # ── LSTM ──────────────────────────────────────────────────
        if LSTM_AVAILABLE and len(df) >= self.cfg.lstm_lookback + 20:
            try:
                self._train_lstm(df)
            except Exception as e:
                log.warning(f"LSTM training failed: {e}")

    def _train_lstm(self, df: pd.DataFrame):
        feat_df = self._get_features(df)
        scaled  = self.scaler.fit_transform(feat_df)
        lb      = self.cfg.lstm_lookback

        X_seq, y_seq = [], []
        y_raw = self._make_target(df).values
        for i in range(lb, len(scaled)-3):
            X_seq.append(scaled[i-lb:i])
            y_seq.append(y_raw[i])
        if not X_seq:
            return

        X_arr = np.array(X_seq)
        y_arr = np.array(y_seq)

        self.lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lb, X_arr.shape[2])),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        self.lstm_model.compile(optimizer="adam", loss="binary_crossentropy",
                                metrics=["accuracy"])
        self.lstm_model.fit(
            X_arr, y_arr, epochs=5, batch_size=32, verbose=0,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
        )
        self.lstm_model.save(os.path.join(self.cfg.model_dir, "lstm_model.keras"))

    def predict(self, df: pd.DataFrame) -> float:
        """Returns probability of up-move (0 to 1). 0.5 = no edge."""
        probs = []

        if self.rf_model is not None:
            try:
                X = self._get_features(df.tail(1))
                probs.append(self.rf_model.predict_proba(X)[0][1])
            except Exception as e:
                log.debug(f"RF predict error: {e}")

        if self.lstm_model is not None and LSTM_AVAILABLE:
            try:
                lb   = self.cfg.lstm_lookback
                feat = self._get_features(df.tail(lb + 1))
                if len(feat) >= lb:
                    scaled = self.scaler.transform(feat)
                    X_seq  = np.array([scaled[-lb:]])
                    probs.append(float(self.lstm_model.predict(X_seq, verbose=0)[0][0]))
            except Exception as e:
                log.debug(f"LSTM predict error: {e}")

        if not probs:
            return 0.5
        # RF weight 0.6, LSTM weight 0.4 (RF more reliable with limited data)
        if len(probs) == 1:
            return probs[0]
        return probs[0] * 0.6 + probs[1] * 0.4

    def load_models(self):
        rf_path   = os.path.join(self.cfg.model_dir, "rf_model.joblib")
        lstm_path = os.path.join(self.cfg.model_dir, "lstm_model.keras")
        if os.path.exists(rf_path) and ML_AVAILABLE:
            try:
                self.rf_model = joblib.load(rf_path)
                log.info("  Loaded saved RF model")
            except Exception: pass
        if os.path.exists(lstm_path) and LSTM_AVAILABLE:
            try:
                self.lstm_model = keras_load(lstm_path)
                log.info("  Loaded saved LSTM model")
            except Exception: pass


# ═══════════════════════════════════════════════════════════════════
#  LAYER 8 │ RISK MANAGER
# ═══════════════════════════════════════════════════════════════════
class RiskManager:
    """Position sizing, drawdown control, trade lifecycle management."""

    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.open_trades  : List[TradePosition] = []
        self.closed_trades: List[TradePosition] = []
        self.daily_pnl    : float = 0.0
        self.daily_reset  : datetime = datetime.now(timezone.utc).replace(
                              hour=0, minute=0, second=0, microsecond=0)

    @property
    def balance(self) -> float:
        return self.cfg.balance + sum(t.pnl for t in self.closed_trades)

    def can_trade(self) -> Tuple[bool, str]:
        # Reset daily PnL at midnight UTC
        now = datetime.now(timezone.utc)
        if now > self.daily_reset + timedelta(days=1):
            self.daily_pnl  = 0.0
            self.daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self.daily_pnl < -self.balance * self.cfg.max_daily_loss:
            return False, f"Daily loss limit hit ({self.daily_pnl:.2f})"
        if len(self.open_trades) >= self.cfg.max_open_trades:
            return False, f"Max open trades ({self.cfg.max_open_trades})"
        return True, "OK"

    def position_size(self, entry: float, stop: float) -> float:
        risk_amt = self.balance * self.cfg.risk_pct
        distance = abs(entry - stop)
        if distance < 1e-9:
            return 0.0
        raw = risk_amt / distance
        # Cap at 10% of balance / current price
        max_qty = (self.balance * 0.10) / entry
        return round(min(raw, max_qty), 6)

    def build_trade(self, signal: SignalResult, symbol: str) -> Optional[TradePosition]:
        ok, reason = self.can_trade()
        if not ok:
            log.info(f"  Trade blocked: {reason}")
            return None
        if signal.entry is None:
            return None
        qty = self.position_size(signal.entry, signal.stop)
        if qty <= 0:
            return None
        return TradePosition(
            symbol  = symbol,
            side    = "BUY" if signal.score > 0 else "SELL",
            entry   = signal.entry,
            stop    = signal.stop,
            tp1     = signal.tp1,
            tp2     = signal.tp2,
            qty     = qty,
            score   = signal.score,
            ai_conf = signal.ai_prob,
        )

    def update_trades(self, current_price: float):
        for trade in self.open_trades:
            if trade.status != "OPEN": continue
            if trade.side == "BUY":
                if current_price <= trade.stop:
                    trade.pnl    = (trade.stop - trade.entry) * trade.qty
                    trade.status = "STOPPED"
                    self.daily_pnl += trade.pnl
                elif not trade.tp1_hit and current_price >= trade.tp1:
                    trade.tp1_hit = True
                    half_pnl = (trade.tp1 - trade.entry) * trade.qty * 0.5
                    trade.pnl += half_pnl
                    self.daily_pnl += half_pnl
                    trade.stop = trade.entry  # move to breakeven
                    log.info(f"  TP1 hit for {trade.symbol} BUY @ {trade.tp1:.0f}")
                elif current_price >= trade.tp2:
                    trade.pnl   += (trade.tp2 - trade.entry) * trade.qty * 0.5
                    trade.status = "CLOSED"
                    self.daily_pnl += trade.pnl
                    log.info(f"  TP2 hit for {trade.symbol} BUY @ {trade.tp2:.0f}")
            else:  # SELL
                if current_price >= trade.stop:
                    trade.pnl    = (trade.entry - trade.stop) * trade.qty
                    trade.status = "STOPPED"
                    self.daily_pnl += trade.pnl
                elif not trade.tp1_hit and current_price <= trade.tp1:
                    trade.tp1_hit = True
                    half_pnl = (trade.entry - trade.tp1) * trade.qty * 0.5
                    trade.pnl += half_pnl
                    self.daily_pnl += half_pnl
                    trade.stop = trade.entry
                    log.info(f"  TP1 hit for {trade.symbol} SELL @ {trade.tp1:.0f}")
                elif current_price <= trade.tp2:
                    trade.pnl   += (trade.entry - trade.tp2) * trade.qty * 0.5
                    trade.status = "CLOSED"
                    self.daily_pnl += trade.pnl

        # Move closed/stopped to history
        done = [t for t in self.open_trades if t.status != "OPEN"]
        for t in done:
            self.closed_trades.append(t)
            self.open_trades.remove(t)


# ═══════════════════════════════════════════════════════════════════
#  LAYER 9 │ SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════
class SignalEngine:
    """Fuses all layers into a composite directional bias score."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(self, df: pd.DataFrame, poc: float, vah: float, val: float,
                wyck_bias: int, fund_bias: int, cvd_net_bull: bool,
                ai_prob: float, session_edge: float,
                near_sh: list, near_sl: list) -> SignalResult:

        last    = df.iloc[-1]
        price   = last["close"]
        atr     = last["atr"]
        score   = 0
        reasons = []

        def add(pts: int, reason: str):
            nonlocal score
            score += pts
            reasons.append(f"{'↑' if pts>0 else '↓' if pts<0 else '○'} {reason:55s} {pts:+d}")

        # ── Order Flow signals ────────────────────────────────────
        if last.get("div_bull", False):    add(+2, "Bullish CVD divergence (price↓ CVD↑)")
        if last.get("div_bear", False):    add(-2, "Bearish CVD divergence (price↑ CVD↓)")
        if last.get("div_score_bull",0)>=2:add(+1, "Double bullish divergence confirmation")
        if last.get("div_score_bear",0)>=2:add(-1, "Double bearish divergence confirmation")

        if last.get("sweep_low", False):   add(+2, "Liquidity sweep LOW (stop hunt ↑)")
        if last.get("sweep_high", False):  add(-2, "Liquidity sweep HIGH (stop hunt ↓)")
        if last.get("sweep_low5", False):  add(+1, "Minor liquidity sweep low")
        if last.get("sweep_high5", False): add(-1, "Minor liquidity sweep high")

        if last.get("stacked_buy", False):  add(+1, "Stacked buy imbalance (3 bars)")
        if last.get("stacked_sell", False): add(-1, "Stacked sell imbalance (3 bars)")
        if last.get("super_stack_buy", False):  add(+1, "Super stacked buy (5 bars)")
        if last.get("super_stack_sell", False): add(-1, "Super stacked sell (5 bars)")

        if last.get("bid_absorb", False):  add(+1, "Bid absorption (sellers absorbed)")
        if last.get("ask_absorb", False):  add(-1, "Ask absorption (buyers absorbed)")
        if last.get("buy_exhaust", False): add(-1, "Buy exhaustion (delta/price mismatch)")
        if last.get("sell_exhaust",False): add(+1, "Sell exhaustion (bears spent)")

        if last.get("trapped_shorts",False):add(+1, "Trapped shorts being squeezed")
        if last.get("trapped_longs", False):add(-1, "Trapped longs being squeezed")

        if last.get("accum_signal", False): add(+1, "Accumulation signal (smart money buying)")
        if last.get("distrib_signal",False):add(-1, "Distribution signal (smart money selling)")

        # ── Institutional / Macro ─────────────────────────────────
        score += wyck_bias
        reasons.append(f"○ Wyckoff phase bias{' '*37}{wyck_bias:+d}")

        score += fund_bias
        reasons.append(f"○ Funding rate bias{' '*38}{fund_bias:+d}")

        if cvd_net_bull: add(+1, "CVD net buying (20-bar)")
        else:            add(-1, "CVD net selling (20-bar)")

        # ── VWAP position ─────────────────────────────────────────
        vwap20_dev = last.get("vwap20_dev", 0) or 0
        if   vwap20_dev > 0.5:  add(-1, f"Extended above VWAP20 (+{vwap20_dev:.2f}%)")
        elif vwap20_dev < -0.5: add(+1, f"Extended below VWAP20 ({vwap20_dev:.2f}%)")

        sess_dev = last.get("sess_vwap_dev", 0) or 0
        if   sess_dev > 0.8:    add(-1, f"Extended above session VWAP (+{sess_dev:.2f}%)")
        elif sess_dev < -0.8:   add(+1, f"Extended below session VWAP ({sess_dev:.2f}%)")

        # ── Value area position ───────────────────────────────────
        if   price > vah: add(+1, "Above Value Area High (acceptance/breakout)")
        elif price < val: add(-1, "Below Value Area Low (acceptance/breakdown)")

        # ── Technical indicators ──────────────────────────────────
        rsi = last.get("rsi", 50) or 50
        if   rsi < 30:   add(+1, f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:   add(-1, f"RSI overbought ({rsi:.0f})")

        macd_h = last.get("macd_hist", 0) or 0
        if   macd_h > 0: add(+1, "MACD histogram positive (bullish momentum)")
        elif macd_h < 0: add(-1, "MACD histogram negative (bearish momentum)")

        if last.get("above_vwap20", last["close"] > last.get("vwap20", last["close"])):
            if last.get("ema_20", 0) and last.get("ema_50", 0):
                if last["ema_20"] > last["ema_50"]:
                    add(+1, "EMA20 > EMA50 (bullish trend)")
                else:
                    add(-1, "EMA20 < EMA50 (bearish trend)")

        # ── Delta / volume ────────────────────────────────────────
        dp = last["delta_pct"]
        if   dp > 0.25: add(+1, f"Strong buy delta ({dp:+.2f})")
        elif dp < -0.25:add(-1, f"Strong sell delta ({dp:+.2f})")

        if last["vol_z"] > 2.5:
            v = +1 if last["is_bull"] else -1
            add(v, f"Volume spike (z={last['vol_z']:.1f})")

        # ── Wick rejection ────────────────────────────────────────
        if last["wick_top"] > atr * 0.35: add(-1, "Top wick rejection")
        if last["wick_bot"] > atr * 0.35: add(+1, "Bottom wick support")

        # ── AI probability ────────────────────────────────────────
        ai_adj = 0
        if   ai_prob > 0.70: ai_adj = +2
        elif ai_prob > 0.60: ai_adj = +1
        elif ai_prob < 0.30: ai_adj = -2
        elif ai_prob < 0.40: ai_adj = -1
        if ai_adj:
            add(ai_adj, f"AI ensemble probability ({ai_prob:.1%})")

        # ── Session edge ──────────────────────────────────────────
        se_pts = round(session_edge)
        if abs(se_pts) >= 1:
            add(se_pts, f"Time-of-day statistical edge ({last['session']})")

        # ── Clamp and classify ────────────────────────────────────
        score      = int(np.clip(score, -10, 10))
        confidence = abs(score) / 10.0

        if   score >=  5: bias = "STRONG LONG ▲▲"
        elif score >=  3: bias = "LONG BIAS   ▲"
        elif score <= -5: bias = "STRONG SHORT ▼▼"
        elif score <= -3: bias = "SHORT BIAS   ▼"
        else:             bias = "NEUTRAL / WAIT ─"

        # ── Trade structure ───────────────────────────────────────
        # Gate: score strong enough AND AI not strongly contradicting
        entry = stop = tp1 = tp2 = rr_val = None
        ai_contradicts = (score > 0 and ai_prob < 0.35) or (score < 0 and ai_prob > 0.65)
        if abs(score) >= self.cfg.min_score and not ai_contradicts:
            if score > 0:  # LONG
                entry = round(price, -1)
                stop  = round(price - atr * self.cfg.atr_stop_mult, -1)
                tp1   = round(price + atr * self.cfg.atr_tp1_mult, -1)
                tp2   = vah if vah > price else round(price + atr * self.cfg.atr_tp2_mult, -1)
            else:           # SHORT
                entry = round(price, -1)
                stop  = round(price + atr * self.cfg.atr_stop_mult, -1)
                tp1   = round(price - atr * self.cfg.atr_tp1_mult, -1)
                tp2   = val if val < price else round(price - atr * self.cfg.atr_tp2_mult, -1)

            if entry and stop and tp1:
                risk  = abs(entry - stop)
                rew   = abs(tp1   - entry)
                rr_val= rew / risk if risk > 0 else 0
                if rr_val < self.cfg.min_rr:
                    entry = stop = tp1 = tp2 = None  # skip poor R:R

        return SignalResult(
            bias=bias, score=score, confidence=confidence,
            ai_prob=ai_prob, reasons=reasons,
            entry=entry, stop=stop, tp1=tp1, tp2=tp2,
            rr=rr_val or 0.0
        )


# ═══════════════════════════════════════════════════════════════════
#  LAYER 10 │ EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════
class ExecutionEngine:
    """Handles order placement, signing, and error recovery."""

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": cfg.api_key,
            "User-Agent": "ApexBot/2.0"
        })

    def _sign(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(self.cfg.api_secret.encode(),
                        query.encode(), hashlib.sha256).hexdigest()

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "MARKET",
                    stop_price: Optional[float] = None) -> dict:
        if self.cfg.mode != "live":
            log.info(f"  [PAPER] {side} {qty:.6f} {symbol} ({order_type})")
            return {"status": "PAPER", "side": side, "qty": qty}

        params = {
            "symbol":    symbol,
            "side":      side,
            "type":      order_type,
            "quantity":  f"{qty:.6f}",
            "timestamp": int(time.time() * 1000),
        }
        if stop_price and order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
            params["stopPrice"] = f"{stop_price:.1f}"

        params["signature"] = self._sign(params)
        try:
            r = self.session.post(
                f"{self.cfg.base_url}/fapi/v1/order",
                params=params, timeout=8
            )
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            log.error(f"Order error: {e} | {r.text}")
            return {"status": "ERROR", "error": str(e)}

    def place_bracket(self, trade: TradePosition) -> dict:
        """Place entry + stop + TP as bracket orders."""
        entry_res = self.place_order(trade.symbol, trade.side, trade.qty)
        if entry_res.get("status") not in ["FILLED", "PAPER"]:
            return entry_res

        stop_side = "SELL" if trade.side == "BUY" else "BUY"
        self.place_order(trade.symbol, stop_side, trade.qty,
                         "STOP_MARKET", trade.stop)
        self.place_order(trade.symbol, stop_side, trade.qty * 0.5,
                         "TAKE_PROFIT_MARKET", trade.tp1)
        self.place_order(trade.symbol, stop_side, trade.qty * 0.5,
                         "TAKE_PROFIT_MARKET", trade.tp2)
        return entry_res


# ═══════════════════════════════════════════════════════════════════
#  LAYER 11 │ ALERT SYSTEM
# ═══════════════════════════════════════════════════════════════════
class AlertSystem:
    """Telegram + console alerts with rate limiting."""

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self._last_alert: Dict[str, datetime] = {}
        self._cooldown  = timedelta(minutes=5)

    def send(self, msg: str, force: bool = False):
        now  = datetime.now(timezone.utc)
        key  = msg[:50]
        if not force and key in self._last_alert:
            if now - self._last_alert[key] < self._cooldown:
                return
        self._last_alert[key] = now

        log.info(f"ALERT: {msg}")

        if self.cfg.telegram and self.cfg.tg_token and self.cfg.tg_chat:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{self.cfg.tg_token}/sendMessage",
                    data={"chat_id": self.cfg.tg_chat, "text": f"🤖 ApexBot\n{msg}",
                          "parse_mode": "Markdown"},
                    timeout=5
                )
            except Exception as e:
                log.warning(f"Telegram send failed: {e}")

    def format_signal(self, sig: SignalResult, price: float, symbol: str) -> str:
        lines = [
            f"*{symbol}* @ ${price:,.0f}",
            f"Bias: {sig.bias}",
            f"Score: {sig.score:+d}/10 | AI: {sig.ai_prob:.1%}",
        ]
        if sig.entry:
            lines += [
                f"Entry: ${sig.entry:,.0f}",
                f"Stop:  ${sig.stop:,.0f}",
                f"TP1:   ${sig.tp1:,.0f}",
                f"TP2:   ${sig.tp2:,.0f}",
                f"R:R:   {sig.rr:.2f}",
            ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════
class ApexBot:
    """Orchestrates all layers into a live / paper / backtest trading engine."""

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.data_eng   = DataEngine(cfg)
        self.feat_forge = FeatureForge()
        self.of_suite   = OrderFlowSuite(cfg)
        self.mkt_struct = MarketStructure()
        self.pat_miner  = PatternMiner()
        self.wyckoff    = WyckoffAnalysis()
        self.ai_ens     = AIEnsemble(cfg)
        self.risk_mgr   = RiskManager(cfg)
        self.sig_eng    = SignalEngine(cfg)
        self.exec_eng   = ExecutionEngine(cfg)
        self.alerts     = AlertSystem(cfg)

        # Live state
        self._dfs       : Dict[str, pd.DataFrame] = {}
        self._funding   : pd.DataFrame = pd.DataFrame()
        self._live_data : Dict[str, deque] = {
            sym: deque(maxlen=2000) for sym in cfg.symbols
        }
        self._candle_count: int = 0

    def load_and_train(self, symbol: str = None):
        sym = symbol or self.cfg.symbol
        self._dfs, self._funding, live = self.data_eng.load_all(sym)

        if not self._dfs:
            log.error("No data loaded — cannot continue")
            return

        prim = "5m" if "5m" in self._dfs else "1m" if "1m" in self._dfs else \
               list(self._dfs.keys())[0]

        df_p = self.feat_forge.build_base(self._dfs[prim])
        df_p = self.of_suite.run_all(df_p)
        df_p = self.mkt_struct.compute_vwap(df_p)
        self._dfs[prim + "_processed"] = df_p

        # Try to load saved models first
        self.ai_ens.load_models()
        if self.ai_ens.rf_model is None:
            self.ai_ens.train(df_p)

        self._edges = self.pat_miner.compute_edges(df_p)
        log.info("  Bot fully initialized ✓")

    def analyze(self, symbol: str = None) -> SignalResult:
        """Full one-shot analysis. Returns SignalResult."""
        sym  = symbol or self.cfg.symbol
        prim = "5m_processed" if "5m_processed" in self._dfs else \
               "5m" if "5m" in self._dfs else list(self._dfs.keys())[0]
        htf  = "1h" if "1h" in self._dfs else prim

        df_p = self._dfs.get(prim)
        df_h = self._dfs.get(htf)
        if df_p is None:
            log.error("No processed data available")
            return SignalResult(bias="NEUTRAL / WAIT", score=0, confidence=0, ai_prob=0.5)

        if df_h is None:
            df_h = df_p

        # Ensure HTF is processed
        if "cvd" not in df_h.columns:
            df_h = self.feat_forge.build_base(df_h)
            df_h = self.of_suite.run_all(df_h)
            df_h = self.mkt_struct.compute_vwap(df_h)

        poc, vah, val = self.mkt_struct.compute_market_profile(df_h)
        near_sh, near_sl, round_lvls = self.mkt_struct.compute_liquidity_map(df_h)
        wyck_bias, fund_bias, cvd_bull = self.wyckoff.analyze(df_p, poc, vah, val, self._funding)
        ai_prob  = self.ai_ens.predict(df_p)
        sess_edge= self.pat_miner.current_session_edge(df_p, self._edges) \
                   if hasattr(self, "_edges") else 0.0

        signal   = self.sig_eng.compute(
            df_p, poc, vah, val, wyck_bias, fund_bias, cvd_bull,
            ai_prob, sess_edge, near_sh, near_sl
        )

        # Print full report
        self._print_report(df_p, poc, vah, val, signal, near_sh, near_sl)
        return signal

    def _print_report(self, df, poc, vah, val, sig: SignalResult,
                      near_sh, near_sl):
        last  = df.iloc[-1]
        price = last["close"]
        atr   = last["atr"]
        W     = 72
        bar   = "═" * W

        print(f"\n{'▓' * W}")
        print(f"  BTC/USDT APEX ENGINE — INSTITUTIONAL ANALYSIS")
        print(f"  Mode: {self.cfg.mode.upper():8s}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'▓' * W}")

        print(f"\n  ── CURRENT STATE ─────────────────────────────────────────────")
        print(f"  Price:        ${price:>12,.1f}")
        print(f"  ATR(14):      ${atr:>12,.1f}  ({atr/price*100:.2f}%)")
        print(f"  Session:      {last['session']}")
        print(f"  RSI:          {last.get('rsi',0):>12.1f}")
        print(f"  MACD hist:    {last.get('macd_hist',0):>+12.3f}")
        print(f"  Vol Z-score:  {last['vol_z']:>+12.2f}")
        print(f"  Delta %:      {last['delta_pct']:>+12.3f}")
        print(f"  CVD slope:    {last.get('cvd_slope3',0):>+12.1f}")

        print(f"\n  ── MARKET STRUCTURE ──────────────────────────────────────────")
        print(f"  POC (most traded):   ${poc:>10,.1f}")
        print(f"  VAH (value area hi): ${vah:>10,.1f}  {'← ABOVE' if price>vah else ''}")
        print(f"  VAL (value area lo): ${val:>10,.1f}  {'← BELOW' if price<val else ''}")
        # Show stops correctly: short stops cluster above price, long stops below
        sh_above = sorted([h for h in near_sh if h > price])
        sl_below = sorted([l for l in near_sl if l < price], reverse=True)
        if sh_above:
            print(f"  Short stops (above): ${sh_above[0]:>10,.0f}  (+${sh_above[0]-price:,.0f})")
        if sl_below:
            print(f"  Long stops  (below): ${sl_below[0]:>10,.0f}  (-${price-sl_below[0]:,.0f})")

        print(f"\n  ── ACTIVE FLAGS ──────────────────────────────────────────────")
        flags = {
            "Bullish CVD divergence": last.get("div_bull", False),
            "Bearish CVD divergence": last.get("div_bear", False),
            "Liquidity sweep LOW":    last.get("sweep_low", False),
            "Liquidity sweep HIGH":   last.get("sweep_high", False),
            "Bid absorption":         last.get("bid_absorb", False),
            "Ask absorption":         last.get("ask_absorb", False),
            "Stacked buy imbalance":  last.get("stacked_buy", False),
            "Stacked sell imbalance": last.get("stacked_sell", False),
            "Whale candle":           last.get("is_whale", False),
            "Iceberg order":          last.get("iceberg", False),
            "Trapped longs":          last.get("trapped_longs", False),
            "Trapped shorts":         last.get("trapped_shorts", False),
            "Buy exhaustion":         last.get("buy_exhaust", False),
            "Sell exhaustion":        last.get("sell_exhaust", False),
        }
        active = {k: v for k, v in flags.items() if v}
        if active:
            for k in active:
                print(f"  ⚡ {k}")
        else:
            print(f"  ○ No strong flags on current candle")

        print(f"\n  ── SIGNAL BREAKDOWN ──────────────────────────────────────────")
        for r in sig.reasons:
            print(f"  {r}")

        print(f"\n  ┌{'─' * 62}┐")
        print(f"  │  BIAS:       {sig.bias:<49}│")
        print(f"  │  Score:      {sig.score:>+4d} / 10   Confidence: {sig.confidence*100:>4.0f}%        │")
        print(f"  │  AI Prob:    {sig.ai_prob:>5.1%}  (RF+LSTM ensemble)              │")
        print(f"  └{'─' * 62}┘")

        if sig.entry:
            risk = abs(sig.entry - sig.stop)
            print(f"\n  ── TRADE STRUCTURE ───────────────────────────────────────────")
            print(f"  Entry:   ${sig.entry:>10,.0f}")
            print(f"  Stop:    ${sig.stop:>10,.0f}  (risk ${risk:,.0f} = {risk/atr:.1f}× ATR)")
            print(f"  TP1:     ${sig.tp1:>10,.0f}  (profit ${abs(sig.tp1-sig.entry):,.0f})")
            print(f"  TP2:     ${sig.tp2:>10,.0f}  (profit ${abs(sig.tp2-sig.entry):,.0f})")
            print(f"  R:R:     {sig.rr:>10.2f}  {'✓ ACCEPTABLE' if sig.rr >= self.cfg.min_rr else '✗ POOR'}")
            qty  = self.risk_mgr.position_size(sig.entry, sig.stop)
            print(f"  Qty:     {qty:>10.6f} BTC  (1% risk = ${self.risk_mgr.balance*0.01:,.0f})")
        else:
            if abs(sig.score) >= self.cfg.min_score:
                print(f"\n  Trade skipped: R:R below minimum {self.cfg.min_rr}")
            else:
                print(f"\n  No trade: Score {sig.score:+d} below threshold ±{self.cfg.min_score}")

        print(f"\n{'▓' * W}\n")

    def process_new_candle(self, candle: dict, symbol: str = None):
        """Called for each new closed candle from WebSocket."""
        sym = symbol or self.cfg.symbol
        self._candle_count += 1

        # Append to live buffer
        buf = self._live_data[sym]
        buf.append(candle)

        if len(buf) < 100:
            return

        # Rebuild dataframe from buffer
        df = pd.DataFrame(list(buf))
        if "open_time" not in df.columns:
            df["open_time"] = pd.date_range(
                end=datetime.now(timezone.utc),
                periods=len(df), freq="1min", tz="UTC"
            )
        if "taker_buy_vol" not in df.columns:
            # Estimate taker buy vol from delta proxy if not provided
            df["taker_buy_vol"] = df["volume"] * (
                df["close"].diff().apply(lambda x: 0.6 if x > 0 else 0.4).fillna(0.5)
            )
        if "trades" not in df.columns:
            df["trades"] = (df["volume"] / 0.05).astype(int)

        df_feat = self.feat_forge.build_base(df)
        df_feat = self.of_suite.run_all(df_feat)
        df_feat = self.mkt_struct.compute_vwap(df_feat)

        # Retrain periodically
        if self._candle_count % self.cfg.retrain_every == 0:
            log.info(f"  Retraining AI models at candle #{self._candle_count}")
            self.ai_ens.train(df_feat)

        # Compute POC/VAH/VAL from HTF
        htf_df = self._dfs.get("1h")
        if htf_df is not None and "atr" not in htf_df.columns:
            htf_df = self.feat_forge.build_base(htf_df)
            htf_df = self.of_suite.run_all(htf_df)
            htf_df = self.mkt_struct.compute_vwap(htf_df)
        poc, vah, val = self.mkt_struct.compute_market_profile(htf_df or df_feat)

        near_sh, near_sl, _ = self.mkt_struct.compute_liquidity_map(df_feat)
        wyck_bias, fund_bias, cvd_bull = self.wyckoff.analyze(
            df_feat, poc, vah, val, self._funding
        )
        ai_prob   = self.ai_ens.predict(df_feat)
        sess_edge = self.pat_miner.current_session_edge(df_feat,
                    getattr(self, "_edges", {}))

        signal    = self.sig_eng.compute(
            df_feat, poc, vah, val, wyck_bias, fund_bias, cvd_bull,
            ai_prob, sess_edge, near_sh, near_sl
        )

        price = df_feat["close"].iloc[-1]
        self.risk_mgr.update_trades(price)

        if signal.entry is not None:
            trade = self.risk_mgr.build_trade(signal, sym)
            if trade:
                result = self.exec_eng.place_bracket(trade)
                self.risk_mgr.open_trades.append(trade)
                alert_msg = self.alerts.format_signal(signal, price, sym)
                self.alerts.send(alert_msg, force=True)

        # Log current state every 10 candles
        if self._candle_count % 10 == 0:
            log.info(f"  {sym} ${price:,.1f}  Score:{signal.score:+d}  "
                     f"AI:{signal.ai_prob:.1%}  "
                     f"Open:{len(self.risk_mgr.open_trades)}  "
                     f"DailyPnL:${self.risk_mgr.daily_pnl:+,.2f}")

    def start_websocket(self):
        """Start WebSocket for live candle stream."""
        def on_message(ws, raw):
            try:
                data = json.loads(raw)
                k    = data.get("k", {})
                if not k.get("x"):  # only closed candles
                    return
                candle = {
                    "open":          float(k["o"]),
                    "high":          float(k["h"]),
                    "low":           float(k["l"]),
                    "close":         float(k["c"]),
                    "volume":        float(k["v"]),
                    "taker_buy_vol": float(k.get("q", k["v"]) * 0.5),
                    "trades":        int(k.get("n", 0)),
                    "open_time":     pd.Timestamp(k["t"], unit="ms", tz="UTC"),
                }
                sym = k.get("s", self.cfg.symbol)
                self.process_new_candle(candle, sym)
            except Exception as e:
                log.error(f"WebSocket message error: {e}")

        def on_error(ws, err):
            log.error(f"WebSocket error: {err}")

        def on_close(ws, *args):
            log.warning("WebSocket closed — reconnecting in 5s…")
            time.sleep(5)
            self.start_websocket()

        def on_open(ws):
            log.info("WebSocket connected ✓")

        streams = "/".join(f"{sym.lower()}@kline_1m" for sym in self.cfg.symbols)
        url     = f"{self.cfg.ws_url}/{streams}"
        log.info(f"  Connecting WebSocket: {url}")

        ws_app  = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error  =on_error,
            on_close  =on_close,
            on_open   =on_open,
        )
        t = threading.Thread(target=ws_app.run_forever, daemon=True)
        t.start()
        return t

    def run(self):
        """Main entry point."""
        print(f"\n{'▓' * 72}")
        print(f"  BTC APEX TRADING ENGINE")
        print(f"  Mode: {self.cfg.mode.upper()}")
        print(f"  Symbols: {', '.join(self.cfg.symbols)}")
        print(f"  Balance: ${self.cfg.balance:,.0f}")
        print(f"  Risk/trade: {self.cfg.risk_pct*100:.1f}%")
        print(f"{'▓' * 72}\n")

        self.load_and_train()

        if self.cfg.mode == "analyze":
            self.analyze()
            return

        self.alerts.send(
            f"ApexBot started\nMode: {self.cfg.mode}\n"
            f"Balance: ${self.risk_mgr.balance:,.0f}", force=True
        )

        ws_thread = self.start_websocket()

        log.info(f"  Running in {self.cfg.mode} mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
                log.info(f"  Balance: ${self.risk_mgr.balance:,.2f}  "
                         f"Open: {len(self.risk_mgr.open_trades)}  "
                         f"Closed: {len(self.risk_mgr.closed_trades)}")
        except KeyboardInterrupt:
            log.info("  Shutting down gracefully…")
            self.alerts.send("ApexBot stopped by user", force=True)


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="BTC Apex Trading Engine")
    p.add_argument("--mode",    default="analyze",
                   choices=["live","paper","backtest","analyze"])
    p.add_argument("--symbol",  default="BTCUSDT")
    p.add_argument("--balance", default=10000, type=float)
    p.add_argument("--risk",    default=0.01, type=float)
    p.add_argument("--api-key",    default="")
    p.add_argument("--api-secret", default="")
    p.add_argument("--telegram",   action="store_true")
    p.add_argument("--tg-token",   default="")
    p.add_argument("--tg-chat",    default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    CFG.mode       = args.mode
    CFG.symbol     = args.symbol
    CFG.symbols    = [args.symbol]
    CFG.balance    = args.balance
    CFG.risk_pct   = args.risk
    if args.api_key:    CFG.api_key    = args.api_key
    if args.api_secret: CFG.api_secret = args.api_secret
    if args.telegram:   CFG.telegram   = True
    if args.tg_token:   CFG.tg_token   = args.tg_token
    if args.tg_chat:    CFG.tg_chat    = args.tg_chat

    bot = ApexBot(CFG)
    bot.run()
