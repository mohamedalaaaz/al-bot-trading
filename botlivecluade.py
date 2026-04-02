#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BTC/USDT INSTITUTIONAL LIVE TRADING BOT                          ║
║           Full Hedge Fund Grade — Binance Futures                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ALL INDICATORS:                                                            ║
║   • CVD Pro          • Big Traders      • Order Flow                       ║
║   • Unfinished Biz   • VWAP/TWAP        • Market Profile                  ║
║   • Footprint        • TPO              • Imbalance Chart                  ║
║   • Liquidity Map    • Wyckoff          • Smart Money                      ║
║   • Funding Intel    • Hurst Regime     • Multi-TF Confluence              ║
║                                                                             ║
║  HEDGE FUND STRATEGIES BAKED IN:                                           ║
║   • Renaissance-style: stat edge + divergence + pattern stacking           ║
║   • Citadel-style: order flow + liquidity sweep + absorption               ║
║   • Two Sigma-style: regime detection + mean reversion vs trend            ║
║   • Paul Tudor Jones: market structure + unfinished business               ║
║   • Stan Druckenmiller: conviction sizing + ride the trend                 ║
║                                                                             ║
║  MODES:                                                                     ║
║   PAPER (default, safe) — simulates all trades, no real money              ║
║   LIVE  (requires API key + --live flag) — real Binance orders             ║
║                                                                             ║
║  INSTALL:                                                                   ║
║   pip install requests pandas numpy websocket-client python-dotenv         ║
║                                                                             ║
║  RUN (paper):  python bot.py                                               ║
║  RUN (live):   python bot.py --live                                        ║
║                                                                             ║
║  CONFIG:  edit CONFIG section below OR create .env file                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, json, hmac, hashlib, threading, argparse, logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from urllib.parse import urlencode

# ── Optional websocket (install: pip install websocket-client) ──
try:
    import websocket
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG  ← Edit these or use .env file
# ══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Binance API (needed for LIVE mode only)
    "API_KEY":    os.getenv("BINANCE_API_KEY",    "YOUR_API_KEY_HERE"),
    "API_SECRET": os.getenv("BINANCE_API_SECRET", "YOUR_API_SECRET_HERE"),

    # ── Symbol & timeframes
    "SYMBOL":     "BTCUSDT",
    "PRIMARY_TF": "5m",        # main signal timeframe
    "HTF":        "1h",        # higher timeframe context
    "LTF":        "1m",        # lower timeframe precision entry

    # ── Risk management
    "ACCOUNT_USDT":    1000.0,   # total account size
    "RISK_PER_TRADE":  0.01,     # 1% risk per trade
    "MAX_POSITIONS":   1,        # max concurrent positions
    "MAX_DAILY_LOSS":  0.03,     # 3% daily loss = stop trading
    "LEVERAGE":        5,        # futures leverage
    "MIN_RR":          1.5,      # minimum risk:reward to enter

    # ── Signal engine thresholds
    "MIN_SCORE":       3,        # minimum composite score to trade
    "MIN_CONFIDENCE":  40,       # minimum confidence %
    "VOL_SPIKE_Z":     3.0,      # z-score to flag big trader
    "IMBALANCE_THR":   3.0,      # bid/ask ratio for imbalance
    "ATR_STOP_MULT":   1.5,      # stop = entry ± ATR × this

    # ── Execution
    "LOOP_SECONDS":    30,       # how often to re-analyze
    "CANDLE_LIMIT":    500,      # candles per TF to fetch
    "LOG_FILE":        "bot_trades.log",
}

BASE_URL = "https://fapi.binance.com"

# ══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("BOT")


# ══════════════════════════════════════════════════════════════════════════
#  BINANCE REST CLIENT
# ══════════════════════════════════════════════════════════════════════════
class BinanceClient:
    def __init__(self, api_key="", api_secret=""):
        self.key    = api_key
        self.secret = api_secret
        self.session= requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        qs  = urlencode(params)
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def get(self, path, params=None, signed=False):
        if params is None: params = {}
        if signed: params = self._sign(params)
        r = self.session.get(f"{BASE_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def post(self, path, params=None):
        if params is None: params = {}
        params = self._sign(params)
        r = self.session.post(f"{BASE_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def delete(self, path, params=None):
        if params is None: params = {}
        params = self._sign(params)
        r = self.session.delete(f"{BASE_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    # ── Market data ──
    def klines(self, symbol, interval, limit=500, start_ms=None):
        p = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms: p["startTime"] = int(start_ms)
        data = self.get("/fapi/v1/klines", p)
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_vol","tbqv","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume","taker_buy_vol","trades"]:
            df[c] = df[c].astype(float)
        return df[["open_time","open","high","low","close","volume","taker_buy_vol","trades"]]

    def funding_rate(self, symbol, limit=100):
        data = self.get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
        df = pd.DataFrame(data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        return df

    def ticker_price(self, symbol):
        return float(self.get("/fapi/v1/ticker/price", {"symbol": symbol})["price"])

    def exchange_info(self, symbol):
        data = self.get("/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            if s["symbol"] == symbol:
                return s
        return {}

    def mark_price(self, symbol):
        return self.get("/fapi/v1/premiumIndex", {"symbol": symbol})

    # ── Account ──
    def account(self):
        return self.get("/fapi/v2/account", signed=True)

    def balance(self):
        data = self.get("/fapi/v2/balance", signed=True)
        for b in data:
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def positions(self, symbol):
        data = self.get("/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        return [p for p in data if float(p["positionAmt"]) != 0]

    def open_orders(self, symbol):
        return self.get("/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    # ── Orders ──
    def set_leverage(self, symbol, leverage):
        return self.post("/fapi/v1/leverage",
                         {"symbol": symbol, "leverage": leverage})

    def set_margin_type(self, symbol, margin_type="ISOLATED"):
        try:
            return self.post("/fapi/v1/marginType",
                             {"symbol": symbol, "marginType": margin_type})
        except: pass  # already set

    def market_order(self, symbol, side, qty):
        return self.post("/fapi/v1/order", {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": f"{qty:.3f}",
        })

    def limit_order(self, symbol, side, qty, price):
        return self.post("/fapi/v1/order", {
            "symbol":      symbol,
            "side":        side,
            "type":        "LIMIT",
            "timeInForce": "GTC",
            "quantity":    f"{qty:.3f}",
            "price":       f"{price:.1f}",
        })

    def stop_order(self, symbol, side, qty, stop_price):
        return self.post("/fapi/v1/order", {
            "symbol":        symbol,
            "side":          side,
            "type":          "STOP_MARKET",
            "quantity":      f"{qty:.3f}",
            "stopPrice":     f"{stop_price:.1f}",
            "closePosition": "true",
        })

    def tp_order(self, symbol, side, qty, stop_price):
        return self.post("/fapi/v1/order", {
            "symbol":        symbol,
            "side":          side,
            "type":          "TAKE_PROFIT_MARKET",
            "quantity":      f"{qty:.3f}",
            "stopPrice":     f"{stop_price:.1f}",
            "closePosition": "true",
        })

    def cancel_all(self, symbol):
        try:
            return self.delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
        except: pass


# ══════════════════════════════════════════════════════════════════════════
#  PAPER TRADING ENGINE
# ══════════════════════════════════════════════════════════════════════════
class PaperEngine:
    def __init__(self, balance=1000.0):
        self.balance     = balance
        self.start_bal   = balance
        self.position    = None   # {"side","entry","qty","sl","tp1","tp2","pnl"}
        self.trades      = []
        self.daily_pnl   = 0.0
        self.total_trades= 0
        self.wins        = 0

    def enter(self, side, entry, sl, tp1, tp2, qty, score, confidence, reason):
        if self.position:
            return False, "Already in position"
        self.position = {
            "side": side, "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "qty": qty,
            "score": score, "confidence": confidence,
            "reason": reason, "time": datetime.now(timezone.utc),
            "tp1_hit": False,
        }
        log.info(f"[PAPER ENTER] {side} @ ${entry:,.1f}  "
                 f"SL=${sl:,.1f}  TP1=${tp1:,.1f}  TP2=${tp2:,.1f}  "
                 f"qty={qty:.3f}  score={score:+d}  conf={confidence:.0f}%")
        return True, "OK"

    def update(self, current_price):
        if not self.position:
            return
        p = self.position
        side  = p["side"]

        # Check TP1 (close 60%)
        if not p["tp1_hit"]:
            if (side == "LONG"  and current_price >= p["tp1"]) or \
               (side == "SHORT" and current_price <= p["tp1"]):
                partial_qty = p["qty"] * 0.6
                pnl = partial_qty * abs(p["tp1"] - p["entry"]) * \
                      (1 if side=="LONG" else -1) * CONFIG["LEVERAGE"]
                self.balance  += pnl
                self.daily_pnl+= pnl
                p["tp1_hit"]   = True
                p["qty"]      *= 0.4   # remaining
                # Move SL to breakeven
                p["sl"] = p["entry"]
                log.info(f"[PAPER TP1] {side} TP1 hit @ ${current_price:,.1f}  "
                         f"PnL=+${pnl:.2f}  SL → breakeven")

        # Check TP2 (remaining 40%)
        if p["tp1_hit"]:
            if (side == "LONG"  and current_price >= p["tp2"]) or \
               (side == "SHORT" and current_price <= p["tp2"]):
                pnl = p["qty"] * abs(p["tp2"] - p["entry"]) * \
                      (1 if side=="LONG" else -1) * CONFIG["LEVERAGE"]
                self.balance  += pnl
                self.daily_pnl+= pnl
                self.wins     += 1
                self.total_trades += 1
                log.info(f"[PAPER TP2] {side} TP2 hit @ ${current_price:,.1f}  "
                         f"PnL=+${pnl:.2f}  ✓ WIN")
                self.trades.append({**p, "exit": current_price, "pnl": pnl, "result": "WIN"})
                self.position = None
                return

        # Check SL
        if (side == "LONG"  and current_price <= p["sl"]) or \
           (side == "SHORT" and current_price >= p["sl"]):
            pnl = p["qty"] * abs(p["sl"] - p["entry"]) * \
                  (-1 if side=="LONG" else 1) * CONFIG["LEVERAGE"]
            self.balance  += pnl
            self.daily_pnl+= pnl
            self.total_trades += 1
            log.info(f"[PAPER SL] {side} stopped @ ${current_price:,.1f}  "
                     f"PnL=${pnl:.2f}  ✗ LOSS")
            self.trades.append({**p, "exit": current_price, "pnl": pnl, "result": "LOSS"})
            self.position = None

    def stats(self):
        wr   = self.wins / self.total_trades * 100 if self.total_trades > 0 else 0
        pnl  = self.balance - self.start_bal
        pnl_pct = pnl / self.start_bal * 100
        return {
            "balance":      self.balance,
            "total_pnl":    pnl,
            "pnl_pct":      pnl_pct,
            "trades":       self.total_trades,
            "wins":         self.wins,
            "win_rate":     wr,
            "daily_pnl":    self.daily_pnl,
            "in_position":  self.position is not None,
        }


# ══════════════════════════════════════════════════════════════════════════
#  LIVE ORDER MANAGER
# ══════════════════════════════════════════════════════════════════════════
class LiveOrderManager:
    def __init__(self, client: BinanceClient, symbol: str):
        self.client   = client
        self.symbol   = symbol
        self.position = None

    def enter(self, side, entry, sl, tp1, tp2, qty, score, confidence, reason):
        try:
            # Set leverage + margin type
            self.client.set_leverage(self.symbol, CONFIG["LEVERAGE"])
            self.client.set_margin_type(self.symbol, "ISOLATED")

            # Market entry
            binance_side = "BUY" if side == "LONG" else "SELL"
            order = self.client.market_order(self.symbol, binance_side, qty)
            log.info(f"[LIVE ENTER] {side} order placed: {order}")

            # Stop loss (opposite side)
            sl_side = "SELL" if side == "LONG" else "BUY"
            self.client.stop_order(self.symbol, sl_side, qty, sl)

            # TP1 (60% qty)
            self.client.tp_order(self.symbol, sl_side, qty*0.6, tp1)

            # TP2 (40% qty)
            self.client.tp_order(self.symbol, sl_side, qty*0.4, tp2)

            self.position = {
                "side": side, "entry": entry, "sl": sl,
                "tp1": tp1, "tp2": tp2, "qty": qty,
                "order_id": order.get("orderId"),
                "time": datetime.now(timezone.utc),
            }
            log.info(f"[LIVE] Position set: {self.position}")
            return True, "OK"
        except Exception as e:
            log.error(f"[LIVE ENTER ERROR] {e}")
            return False, str(e)

    def check_position(self):
        try:
            positions = self.client.positions(self.symbol)
            self.position = positions[0] if positions else None
        except Exception as e:
            log.error(f"[POSITION CHECK ERROR] {e}")

    def close_all(self):
        try:
            self.client.cancel_all(self.symbol)
            positions = self.client.positions(self.symbol)
            for p in positions:
                amt  = float(p["positionAmt"])
                side = "SELL" if amt > 0 else "BUY"
                self.client.market_order(self.symbol, side, abs(amt))
            log.info("[LIVE] All positions closed")
        except Exception as e:
            log.error(f"[CLOSE ERROR] {e}")


# ══════════════════════════════════════════════════════════════════════════
#  INDICATOR ENGINE  (all 13 modules)
# ══════════════════════════════════════════════════════════════════════════
class IndicatorEngine:

    @staticmethod
    def base_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["body"]       = d["close"] - d["open"]
        d["body_pct"]   = d["body"] / d["open"] * 100
        d["is_bull"]    = d["body"] > 0
        d["range"]      = d["high"] - d["low"]
        d["wick_top"]   = d["high"] - d[["open","close"]].max(axis=1)
        d["wick_bot"]   = d[["open","close"]].min(axis=1) - d["low"]
        d["sell_vol"]   = d["volume"] - d["taker_buy_vol"]
        d["delta"]      = d["taker_buy_vol"] - d["sell_vol"]
        d["delta_pct"]  = (d["delta"] / d["volume"].replace(0, np.nan)).fillna(0)

        hl  = d["high"] - d["low"]
        hpc = (d["high"] - d["close"].shift(1)).abs()
        lpc = (d["low"]  - d["close"].shift(1)).abs()
        d["atr"] = pd.concat([hl,hpc,lpc], axis=1).max(axis=1).rolling(14).mean()

        roll_mean = d["volume"].rolling(50).mean()
        roll_std  = d["volume"].rolling(50).std()
        d["vol_z"] = (d["volume"] - roll_mean) / roll_std.replace(0, np.nan)

        d["hour"]    = d["open_time"].dt.hour
        d["dow"]     = d["open_time"].dt.day_name()
        d["session"] = d["hour"].apply(
            lambda h: "Asia" if h < 8 else
                      "London" if h < 13 else
                      "NY" if h < 20 else "Late"
        )
        return d.fillna(0)

    @staticmethod
    def cvd_pro(d: pd.DataFrame) -> dict:
        cvd_roll  = d["delta"].rolling(20).sum()
        cvd_slope = cvd_roll.diff(3)
        pr_slope  = d["close"].diff(3) / d["close"].shift(3) * 100

        div_bull  = (pr_slope < -0.15) & (cvd_slope > 0)
        div_bear  = (pr_slope >  0.15) & (cvd_slope < 0)
        exhaustion= (d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.08)
        absorption= (d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.08) & \
                    (d["wick_bot"] + d["wick_top"] > d["atr"] * 0.3)

        last = len(d) - 1
        return {
            "cvd_value":   float(cvd_roll.iloc[last]),
            "cvd_slope":   float(cvd_slope.iloc[last]),
            "price_slope": float(pr_slope.iloc[last]),
            "div_bull":    bool(div_bull.iloc[last]),
            "div_bear":    bool(div_bear.iloc[last]),
            "exhaustion":  bool(exhaustion.iloc[last]),
            "absorption":  bool(absorption.iloc[last]),
            "total_bull_divs": int(div_bull.sum()),
            "total_bear_divs": int(div_bear.sum()),
            "cvd_net_20":  float(cvd_roll.iloc[last] - cvd_roll.iloc[max(0,last-20)]),
            "signal_score": 2 if div_bull.iloc[last] else (-2 if div_bear.iloc[last] else 0),
        }

    @staticmethod
    def big_traders(d: pd.DataFrame) -> dict:
        vol_z     = d["vol_z"]
        is_whale  = vol_z > CONFIG["VOL_SPIKE_Z"]
        accum     = (~d["is_bull"]) & (d["delta_pct"] > 0.15) & (vol_z > 0.5)
        distrib   = d["is_bull"]  & (d["delta_pct"] < -0.15) & (vol_z > 0.5)
        spoof     = (d["is_bull"] & (d["delta_pct"] < -0.3)) | \
                    (~d["is_bull"] & (d["delta_pct"] > 0.3))
        last = len(d) - 1

        # Recent whale direction
        recent_whales = d[is_whale].tail(5)
        whale_bull_pct = recent_whales["is_bull"].mean() if len(recent_whales) > 0 else 0.5

        score = 0
        if is_whale.iloc[last]:
            score = 1 if d["is_bull"].iloc[last] else -1
        if accum.iloc[last]: score += 1
        if distrib.iloc[last]: score -= 1

        return {
            "is_whale":       bool(is_whale.iloc[last]),
            "whale_count":    int(is_whale.sum()),
            "accum_signal":   bool(accum.iloc[last]),
            "distrib_signal": bool(distrib.iloc[last]),
            "spoof_suspect":  bool(spoof.iloc[last]),
            "whale_bull_pct": float(whale_bull_pct),
            "signal_score":   score,
        }

    @staticmethod
    def order_flow(d: pd.DataFrame) -> dict:
        delta_pos    = d["delta_pct"] > 0.1
        stacked_buy  = delta_pos.rolling(3).sum() == 3
        stacked_sell = (~delta_pos).rolling(3).sum() == 3
        bid_absorb   = (d["wick_bot"] > d["atr"]*0.3) & \
                       (d["delta_pct"] > 0.1) & (d["vol_z"] > 1.0)
        ask_absorb   = (d["wick_top"] > d["atr"]*0.3) & \
                       (d["delta_pct"] < -0.1) & (d["vol_z"] > 1.0)
        buy_exhaust  = (d["delta_pct"] > 0.3) & (d["body_pct"].abs() < 0.05)
        sell_exhaust = (d["delta_pct"] < -0.3) & (d["body_pct"].abs() < 0.05)
        trapped_long = (d["body_pct"].shift(1) > 0.25) & \
                       (d["close"] < d["open"].shift(1))
        trapped_short= (d["body_pct"].shift(1) < -0.25) & \
                       (d["close"] > d["open"].shift(1))
        last = len(d) - 1

        score = 0
        if stacked_buy.iloc[last]:   score += 2
        if stacked_sell.iloc[last]:  score -= 2
        if bid_absorb.iloc[last]:    score += 1
        if ask_absorb.iloc[last]:    score -= 1
        if buy_exhaust.iloc[last]:   score -= 1
        if sell_exhaust.iloc[last]:  score += 1
        if trapped_long.iloc[last]:  score -= 1
        if trapped_short.iloc[last]: score += 1

        return {
            "stacked_buy":   bool(stacked_buy.iloc[last]),
            "stacked_sell":  bool(stacked_sell.iloc[last]),
            "bid_absorb":    bool(bid_absorb.iloc[last]),
            "ask_absorb":    bool(ask_absorb.iloc[last]),
            "buy_exhaust":   bool(buy_exhaust.iloc[last]),
            "sell_exhaust":  bool(sell_exhaust.iloc[last]),
            "trapped_long":  bool(trapped_long.iloc[last]),
            "trapped_short": bool(trapped_short.iloc[last]),
            "signal_score":  score,
        }

    @staticmethod
    def vwap_twap(d: pd.DataFrame) -> dict:
        tp = (d["high"] + d["low"] + d["close"]) / 3
        for n in [20, 50]:
            d[f"vwap{n}"] = (tp*d["volume"]).rolling(n).sum() \
                           / d["volume"].rolling(n).sum()
        d["twap20"] = tp.rolling(20).mean()

        # Session VWAP
        d["date"] = d["open_time"].dt.date
        d["sess_vwap"] = (tp*d["volume"]).groupby(d["date"]).cumsum() \
                       / d["volume"].groupby(d["date"]).cumsum()

        # VWAP bands
        var = (d["volume"] * (tp - d["vwap20"])**2).rolling(20).sum() \
            / d["volume"].rolling(20).sum().replace(0, np.nan)
        d["vwap_std"] = np.sqrt(var)
        d["vwap_u1"]  = d["vwap20"] + d["vwap_std"]
        d["vwap_l1"]  = d["vwap20"] - d["vwap_std"]
        d["vwap_u2"]  = d["vwap20"] + 2*d["vwap_std"]
        d["vwap_l2"]  = d["vwap20"] - 2*d["vwap_std"]

        last  = d.iloc[-1]
        price = last["close"]
        dev20 = (price - last["vwap20"]) / last["vwap20"] * 100
        devS  = (price - last["sess_vwap"]) / last["sess_vwap"] * 100 \
                if last["sess_vwap"] > 0 else 0

        score = 0
        if dev20 < -0.3:  score += 1
        if dev20 >  0.3:  score -= 1
        if price <= last["vwap_l2"]: score += 2
        if price >= last["vwap_u2"]: score -= 2

        return {
            "vwap20":    float(last["vwap20"]),
            "vwap50":    float(last["vwap50"]),
            "twap20":    float(last["twap20"]),
            "sess_vwap": float(last["sess_vwap"]),
            "vwap_u1":   float(last["vwap_u1"]),
            "vwap_l1":   float(last["vwap_l1"]),
            "vwap_u2":   float(last["vwap_u2"]),
            "vwap_l2":   float(last["vwap_l2"]),
            "dev_20pct": float(dev20),
            "dev_sess":  float(devS),
            "above_vwap20": price > last["vwap20"],
            "at_upper_band": price >= last["vwap_u1"],
            "at_lower_band": price <= last["vwap_l1"],
            "signal_score":  score,
        }

    @staticmethod
    def market_profile(d: pd.DataFrame, tick=25.0) -> dict:
        lo, hi  = d["low"].min(), d["high"].max()
        buckets = np.arange(np.floor(lo/tick)*tick,
                            np.ceil(hi/tick)*tick + tick, tick)
        vol_map, buy_map, sell_map = defaultdict(float), defaultdict(float), defaultdict(float)
        for _, row in d.iterrows():
            lvls = buckets[(buckets >= row["low"]) & (buckets <= row["high"])]
            if not len(lvls): continue
            v = row["volume"] / len(lvls)
            b = row["taker_buy_vol"] / len(lvls)
            s = row["sell_vol"] / len(lvls)
            for l in lvls:
                vol_map[l] += v; buy_map[l] += b; sell_map[l] += s

        if not vol_map:
            return {"poc":0,"vah":0,"val":0,"signal_score":0}

        profile = pd.DataFrame({
            "price": list(vol_map.keys()),
            "vol":   list(vol_map.values()),
        }).sort_values("price")
        total  = profile["vol"].sum()
        profile["pct"] = profile["vol"] / total

        poc = profile.loc[profile["vol"].idxmax(), "price"]
        # Value area 70%
        poc_i   = profile["vol"].idxmax()
        cum_vol = 0
        va_rows = [poc_i]
        while cum_vol/total < 0.70:
            ui = max(va_rows)+1
            li = min(va_rows)-1
            uv = profile.loc[ui,"vol"] if ui in profile.index else 0
            dv = profile.loc[li,"vol"] if li in profile.index else 0
            if uv >= dv and ui in profile.index:
                va_rows.append(ui); cum_vol += uv
            elif li in profile.index:
                va_rows.append(li); cum_vol += dv
            else: break

        vah = profile.loc[va_rows,"price"].max()
        val = profile.loc[va_rows,"price"].min()
        price = d["close"].iloc[-1]

        score = 0
        if price > vah: score += 1    # breakout territory
        if price < val: score -= 1    # breakdown territory
        if price < poc: score -= 1    # below value center
        if price > poc: score += 1

        return {
            "poc": float(poc), "vah": float(vah), "val": float(val),
            "price_vs_poc": float(price - poc),
            "above_vah": price > vah,
            "below_val": price < val,
            "inside_va": val <= price <= vah,
            "signal_score": score,
        }

    @staticmethod
    def tpo(d: pd.DataFrame, tick=25.0, period_min=30) -> dict:
        d2 = d.copy()
        d2["period"] = d2["open_time"].dt.floor(f"{period_min}min")
        from collections import Counter
        tpo_count = Counter()
        periods   = []
        for period, grp in d2.groupby("period"):
            for _, row in grp.iterrows():
                lvls = np.arange(np.floor(row["low"]/tick)*tick,
                                 np.ceil(row["high"]/tick)*tick, tick)
                for l in lvls:
                    tpo_count[l] += 1
            prices_in = set()
            for _, row in grp.iterrows():
                for l in np.arange(np.floor(row["low"]/tick)*tick,
                                   np.ceil(row["high"]/tick)*tick, tick):
                    prices_in.add(l)
            periods.append(prices_in)

        poc = max(tpo_count, key=tpo_count.get) if tpo_count else 0
        ib_high = max(periods[0] | periods[1]) if len(periods)>=2 else 0
        ib_low  = min(periods[0] | periods[1]) if len(periods)>=2 else 0
        singles = [p for p,c in tpo_count.items() if c == 1]
        price   = d["close"].iloc[-1]

        # Is price in single print zone?
        in_single = any(abs(price-s) < tick for s in singles)
        near_ib   = abs(price-ib_high) < tick*3 or abs(price-ib_low) < tick*3

        score = 0
        if in_single: score += 1   # fast move zone — follow direction
        if near_ib:   score += 1 if price > (ib_high+ib_low)/2 else -1

        return {
            "tpo_poc":    float(poc),
            "ib_high":    float(ib_high),
            "ib_low":     float(ib_low),
            "ib_range":   float(ib_high - ib_low),
            "single_prints": len(singles),
            "in_single_print": in_single,
            "near_ib_extreme": near_ib,
            "signal_score": score,
        }

    @staticmethod
    def imbalance(d: pd.DataFrame, tick=25.0, n=30) -> dict:
        d2 = d.tail(n).copy()
        price = d2["close"].iloc[-1]
        buy_z, sell_z = defaultdict(float), defaultdict(float)
        for _, row in d2.iterrows():
            lvls = np.arange(np.floor(row["low"]/tick)*tick,
                             np.ceil(row["high"]/tick)*tick, tick)
            if not len(lvls): continue
            bv = row["taker_buy_vol"] / len(lvls)
            sv = row["sell_vol"] / len(lvls)
            for l in lvls:
                ratio = bv/sv if sv > 0.01 else 10
                if ratio >= CONFIG["IMBALANCE_THR"]:   buy_z[l]  += bv
                elif ratio <= 1/CONFIG["IMBALANCE_THR"]: sell_z[l] += sv

        # Find stacked zones near price
        thr = tick * 10
        near_buy_z  = sum(v for k,v in buy_z.items()  if abs(k-price) < thr)
        near_sell_z = sum(v for k,v in sell_z.items() if abs(k-price) < thr)

        score = 0
        if near_buy_z > near_sell_z * 1.5:  score += 1
        if near_sell_z > near_buy_z * 1.5:  score -= 1

        return {
            "buy_zones":    len(buy_z),
            "sell_zones":   len(sell_z),
            "near_buy_vol": float(near_buy_z),
            "near_sell_vol":float(near_sell_z),
            "buy_dominant": near_buy_z > near_sell_z,
            "signal_score": score,
        }

    @staticmethod
    def liquidity_map(d: pd.DataFrame) -> dict:
        price = d["close"].iloc[-1]
        atr   = d["atr"].iloc[-1]
        tol   = atr * 0.3

        # Equal highs = short stops above
        eq_highs = [round(d["high"].iloc[i], -2)
                    for i in range(5, len(d)-2)
                    if ((d["high"].iloc[max(0,i-10):i] - d["high"].iloc[i]).abs() < tol).any()]
        # Equal lows = long stops below
        eq_lows  = [round(d["low"].iloc[i], -2)
                    for i in range(5, len(d)-2)
                    if ((d["low"].iloc[max(0,i-10):i] - d["low"].iloc[i]).abs() < tol).any()]

        from collections import Counter
        top_stops_above = [k for k,_ in Counter(eq_highs).most_common(3) if k > price]
        top_stops_below = [k for k,_ in Counter(eq_lows).most_common(3)  if k < price]

        # Nearest round numbers
        rnd = [round(price/n)*n for n in [100,250,500,1000]]
        rnd = sorted(set([r for r in rnd if r != 0]))

        # Nearest above/below
        above_stop = min([s for s in top_stops_above], default=price + atr*10)
        below_stop = max([s for s in top_stops_below], default=price - atr*10)

        # Liquidity grab probability
        dist_up = above_stop - price
        dist_dn = price - below_stop
        likely_grab = "UP" if dist_up < dist_dn * 0.7 else \
                      "DOWN" if dist_dn < dist_up * 0.7 else "NEUTRAL"

        score = 0
        if likely_grab == "UP":   score += 1   # price likely sweeps up first
        if likely_grab == "DOWN": score -= 1

        return {
            "stops_above":    top_stops_above,
            "stops_below":    top_stops_below,
            "nearest_above":  float(above_stop),
            "nearest_below":  float(below_stop),
            "dist_to_above":  float(dist_up),
            "dist_to_below":  float(dist_dn),
            "likely_grab":    likely_grab,
            "round_levels":   rnd,
            "signal_score":   score,
        }

    @staticmethod
    def unfinished_business(d: pd.DataFrame) -> dict:
        price  = d["close"].iloc[-1]
        # Unfilled gaps
        gaps = []
        for i in range(1, len(d)):
            ph, pl = d["high"].iloc[i-1], d["low"].iloc[i-1]
            co = d["open"].iloc[i]
            if co > ph * 1.0008:
                if d["low"].iloc[i:].min() > ph:   # still unfilled
                    gaps.append({"dir": "UP",  "level": ph, "dist": ph - price})
            elif co < pl * 0.9992:
                if d["high"].iloc[i:].max() < pl:  # still unfilled
                    gaps.append({"dir": "DOWN","level": pl, "dist": pl - price})

        gaps_near = sorted(gaps, key=lambda g: abs(g["dist"]))[:5]

        # Naked vol nodes
        thresh = d["volume"].quantile(0.96)
        nodes  = []
        for i, row in d[d["volume"] >= thresh].iterrows():
            lev = (row["high"] + row["low"]) / 2
            fut = d[d["open_time"] > row["open_time"]]
            if fut.empty: continue
            if not ((fut["low"]<=lev)&(fut["high"]>=lev)).any():
                nodes.append({"level": lev, "dist": lev-price, "bull": row["is_bull"]})
        nodes_near = sorted(nodes, key=lambda n: abs(n["dist"]))[:3]

        # Pull toward nearest unfilled gap
        pull_bias = 0
        if gaps_near:
            g = gaps_near[0]
            if abs(g["dist"]) < d["atr"].iloc[-1] * 5:
                pull_bias = 1 if g["dist"] > 0 else -1

        return {
            "unfilled_gaps":   len(gaps),
            "nearest_gaps":    gaps_near,
            "naked_nodes":     len(nodes),
            "nearest_nodes":   nodes_near,
            "pull_bias":       pull_bias,
            "signal_score":    pull_bias,
        }

    @staticmethod
    def wyckoff_hedge(d: pd.DataFrame, funding_df: pd.DataFrame) -> dict:
        """All hedge fund layers: Wyckoff, smart money, funding, regime, Druckenmiller."""
        recent = d.tail(30)
        n = len(recent)
        x = np.arange(n)

        price_slope   = np.polyfit(x, recent["close"].values, 1)[0]
        buy_slope     = np.polyfit(x, recent["taker_buy_vol"].values, 1)[0]
        sell_slope    = np.polyfit(x, recent["sell_vol"].values, 1)[0]

        # Wyckoff phase
        if price_slope < -0.3 and buy_slope > 0:
            wyckoff, w_score = "ACCUMULATION", 2
        elif price_slope > 0.3 and sell_slope > 0:
            wyckoff, w_score = "DISTRIBUTION", -2
        elif price_slope < -0.3 and sell_slope > 0:
            wyckoff, w_score = "MARKDOWN", -2
        elif price_slope > 0.3 and buy_slope > 0:
            wyckoff, w_score = "MARKUP", 2
        else:
            wyckoff, w_score = "CONSOLIDATION", 0

        # Market regime (Hurst-like)
        std10 = recent["close"].rolling(10).std().mean()
        std30 = recent["close"].rolling(30).std().mean() if n >= 30 else std10
        hurst = std10 / std30 if std30 > 0 else 0.5
        if   hurst > 0.65: regime, r_score = "TRENDING",      0  # follow trend
        elif hurst < 0.40: regime, r_score = "MEAN_REVERTING",-1  # fade extremes
        else:              regime, r_score = "MIXED",          0

        # Two Sigma regime overlay
        vol_ratio = recent["volume"].tail(5).mean() / recent["volume"].mean()
        if vol_ratio > 1.5 and abs(price_slope) > 1.0:
            regime += "+HIGH_VOL_TREND"

        # Funding rate (PTJ style — don't fight the crowd forever)
        fund_score = 0
        fund_signal = "Neutral"
        if not funding_df.empty:
            avg_fr = funding_df["fundingRate"].tail(8).mean()
            if avg_fr > 0.0005:
                fund_signal, fund_score = "LONG_OVERHEATED→short", -1
            elif avg_fr < -0.0003:
                fund_signal, fund_score = "SHORT_OVERHEATED→long", 1
            else:
                fund_signal = f"Neutral ({avg_fr*100:.4f}%)"

        # CVD net flow (smart money accumulation signal)
        cvd_20 = d["delta"].rolling(20).sum()
        cvd_trend = cvd_20.iloc[-1] - cvd_20.iloc[-20] if len(cvd_20) >= 20 else 0
        sm_score  = 1 if cvd_trend > 0 else -1

        # Druckenmiller conviction: if Wyckoff + CVD + session all agree → size up
        session_ok = d["session"].iloc[-1] in ["London", "NY"]
        conviction_size = 1.5 if (abs(w_score) >= 2 and sm_score == np.sign(w_score)
                                  and session_ok) else 1.0

        total_score = w_score + fund_score + sm_score

        return {
            "wyckoff":         wyckoff,
            "regime":          regime,
            "hurst":           float(hurst),
            "fund_signal":     fund_signal,
            "cvd_trend":       float(cvd_trend),
            "conviction_mult": conviction_size,
            "w_score":         w_score,
            "fund_score":      fund_score,
            "sm_score":        sm_score,
            "signal_score":    total_score,
        }

    @staticmethod
    def footprint_score(d: pd.DataFrame, n=5) -> dict:
        """Score recent footprint: bid/ask pressure at current price."""
        last_n = d.tail(n)
        buy_pressure  = last_n["taker_buy_vol"].sum()
        sell_pressure = last_n["sell_vol"].sum()
        total         = buy_pressure + sell_pressure
        buy_ratio     = buy_pressure / total if total > 0 else 0.5
        delta_sum     = last_n["delta"].sum()

        # Footprint score
        score = 0
        if buy_ratio > 0.6:  score += 1
        if buy_ratio < 0.4:  score -= 1
        if delta_sum > 0:    score += 1
        if delta_sum < 0:    score -= 1

        return {
            "buy_ratio":    float(buy_ratio),
            "delta_sum":    float(delta_sum),
            "buy_pressure": float(buy_pressure),
            "sell_pressure":float(sell_pressure),
            "signal_score": score,
        }


# ══════════════════════════════════════════════════════════════════════════
#  COMPOSITE SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════
class SignalEngine:
    def compute(self, results: dict, price: float, atr: float,
                poc: float, vah: float, val: float) -> dict:

        # Aggregate all module scores
        score = sum([
            results["cvd"]["signal_score"],
            results["big_traders"]["signal_score"],
            results["order_flow"]["signal_score"],
            results["vwap"]["signal_score"],
            results["market_profile"]["signal_score"],
            results["tpo"]["signal_score"],
            results["imbalance"]["signal_score"],
            results["liquidity"]["signal_score"],
            results["unfinished"]["signal_score"],
            results["hedge_fund"]["signal_score"],
            results["footprint"]["signal_score"],
        ])

        # Multi-TF confluence bonus
        if "htf" in results and "ltf" in results:
            htf_score = results["htf"].get("composite_score", 0)
            ltf_score = results["ltf"].get("composite_score", 0)
            if np.sign(htf_score) == np.sign(score) and abs(htf_score) >= 2:
                score += 1    # HTF agrees
            if np.sign(ltf_score) == np.sign(score):
                score += 1    # LTF confirms

        score = int(np.clip(score, -15, 15))
        confidence = abs(score) / 15 * 100
        conv_mult  = results["hedge_fund"].get("conviction_mult", 1.0)

        # Bias
        if   score >= 5:  bias = "STRONG LONG ▲▲"
        elif score >= 3:  bias = "LONG ▲"
        elif score <= -5: bias = "STRONG SHORT ▼▼"
        elif score <= -3: bias = "SHORT ▼"
        else:             bias = "NO TRADE (noise)"

        # Position sizing (Druckenmiller: size up on conviction)
        risk_usdt   = CONFIG["ACCOUNT_USDT"] * CONFIG["RISK_PER_TRADE"] * conv_mult
        stop_dist   = atr * CONFIG["ATR_STOP_MULT"]
        qty         = (risk_usdt / stop_dist) if stop_dist > 0 else 0

        # Trade levels
        if score >= CONFIG["MIN_SCORE"]:
            side = "LONG"
            sl   = round(price - stop_dist, 1)
            tp1  = round(price + stop_dist * 2, 1)
            tp2  = round(price + stop_dist * 4, 1)
            # Use structure for TP
            if vah > price: tp2 = max(tp2, vah)
        elif score <= -CONFIG["MIN_SCORE"]:
            side = "SHORT"
            sl   = round(price + stop_dist, 1)
            tp1  = round(price - stop_dist * 2, 1)
            tp2  = round(price - stop_dist * 4, 1)
            if val < price: tp2 = min(tp2, val)
        else:
            side = sl = tp1 = tp2 = None

        rr = abs(tp1 - price) / stop_dist if tp1 and stop_dist > 0 else 0

        # Build reason string
        active = []
        if results["cvd"]["div_bull"]:           active.append("CVD_bullDiv")
        if results["cvd"]["div_bear"]:           active.append("CVD_bearDiv")
        if results["cvd"]["absorption"]:         active.append("Absorption")
        if results["big_traders"]["is_whale"]:   active.append("WhaleCandle")
        if results["big_traders"]["accum_signal"]:active.append("Accumulation")
        if results["order_flow"]["stacked_buy"]: active.append("StackedBuy")
        if results["order_flow"]["stacked_sell"]:active.append("StackedSell")
        if results["order_flow"]["bid_absorb"]:  active.append("BidAbsorb")
        if results["order_flow"]["buy_exhaust"]: active.append("BuyExhaust")
        if results["vwap"]["at_lower_band"]:     active.append("VWAP_-2σ")
        if results["vwap"]["at_upper_band"]:     active.append("VWAP_+2σ")
        if results["tpo"]["in_single_print"]:    active.append("SinglePrint")
        if results["unfinished"]["pull_bias"] != 0: active.append("UnfinishedBiz")
        wyck = results["hedge_fund"]["wyckoff"]
        if wyck in ["ACCUMULATION","MARKUP"]:    active.append(f"Wyckoff:{wyck}")
        if wyck in ["DISTRIBUTION","MARKDOWN"]:  active.append(f"Wyckoff:{wyck}")
        fund = results["hedge_fund"]["fund_signal"]
        if "OVERHEATED" in fund:                 active.append(f"Funding:{fund[:10]}")
        liq = results["liquidity"]["likely_grab"]
        if liq != "NEUTRAL":                     active.append(f"LiqGrab:{liq}")

        reason = " | ".join(active) or "composite"

        return {
            "score":       score,
            "confidence":  confidence,
            "bias":        bias,
            "side":        side,
            "sl":          sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "qty":         round(qty, 3),
            "rr":          rr,
            "conv_mult":   conv_mult,
            "reason":      reason,
            "tradeable":   (side is not None and
                           confidence >= CONFIG["MIN_CONFIDENCE"] and
                           rr >= CONFIG["MIN_RR"]),
        }



# Install
pip install requests pandas numpy websocket-client

# Paper mode (safe, default)
python bot.py

# Custom settings
python bot.py --account 5000 --risk 0.01 --leverage 10 --interval 1m

# Backtest signal edge first
python bot.py --backtest

# Live trading (needs API keys in .env)
BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python bot.py --live