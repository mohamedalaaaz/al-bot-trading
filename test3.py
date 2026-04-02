"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       BTC/USDT BINANCE FUTURES — INSTITUTIONAL ANALYSIS ENGINE              ║
║       Full Order Flow Intelligence Suite                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODULE 01 │ DATA ENGINE          — Binance Futures multi-TF fetch          ║
║  MODULE 02 │ CVD PRO              — Delta, CVD divergence, exhaustion       ║
║  MODULE 03 │ BIG TRADERS          — Iceberg detection, whale prints         ║
║  MODULE 04 │ ORDER FLOW           — Absorption, imbalance, stacked bids     ║
║  MODULE 05 │ UNFINISHED BUSINESS  — Naked POC, gaps, single prints          ║
║  MODULE 06 │ VWAP / TWAP          — Session, anchored, deviation bands      ║
║  MODULE 07 │ MARKET PROFILE       — Value area, POC, distribution shape     ║
║  MODULE 08 │ FOOTPRINT            — Price ladder bid/ask per level          ║
║  MODULE 09 │ TPO ANALYSIS         — IB, extensions, single prints           ║
║  MODULE 10 │ IMBALANCE CHART      — Stacked bid/ask imbalances              ║
║  MODULE 11 │ LIQUIDITY MAP        — Stop clusters, liquidation zones        ║
║  MODULE 12 │ HEDGE FUND LAYER     — Institutional smart money analysis      ║
║  MODULE 13 │ SIGNAL ENGINE        — Final trade bias + confidence score     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════
SYMBOL        = "BTCUSDT"
BASE_URL      = "https://fapi.binance.com"
PRIMARY_TF    = "5m"
ANALYSIS_TF   = "1h"
LOOKBACK_1M   = 3000
LOOKBACK_5M   = 2000
LOOKBACK_1H   = 1000
IMBALANCE_THR = 3.0    # bid/ask ratio to flag imbalance
BIG_TRADE_X   = 5.0    # vol z-score to flag big trader
LIQUIDITY_ATR = 1.5    # ATR multiples for stop cluster radius
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════
#  MODULE 01 │ DATA ENGINE
# ══════════════════════════════════════════════════════════════════
def fetch_klines(symbol, interval, limit=1500, start_ms=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = int(start_ms)
    r = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=12)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_vol","tbqv","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","taker_buy_vol","trades"]:
        df[c] = df[c].astype(float)
    return df.drop(columns=["close_time","quote_vol","tbqv","ignore"])

def fetch_multi_batch(symbol, interval, target=3000):
    mins = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240}.get(interval,60)
    start = datetime.now(timezone.utc) - timedelta(minutes=target*mins)
    all_dfs, fetched = [], 0
    cur_ms = start.timestamp() * 1000
    while fetched < target:
        df = fetch_klines(symbol, interval, 1500, cur_ms)
        if df.empty: break
        all_dfs.append(df)
        fetched += len(df)
        cur_ms = df["open_time"].iloc[-1].timestamp()*1000 + mins*60000
        if len(df) < 1500: break
    if not all_dfs: return pd.DataFrame()
    out = pd.concat(all_dfs).drop_duplicates("open_time").sort_values("open_time")
    return out.reset_index(drop=True)

def fetch_funding(symbol, limit=500):
    r = requests.get(f"{BASE_URL}/fapi/v1/fundingRate",
                     params={"symbol":symbol,"limit":limit}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df

# ── Synthetic fallback (mirrors your chart structure: 66k–68.5k range) ──
def synthetic_klines(n, interval_min, base_price=67000, seed=0):
    np.random.seed(seed)
    dates = pd.date_range(
        end=datetime.now(timezone.utc).replace(second=0, microsecond=0),
        periods=n, freq=f"{interval_min}min", tz="UTC"
    )
    price = float(base_price)
    rows = []
    for dt in dates:
        h = dt.hour
        sv = 2.2 if h in [8,9,13,14,15,16] else (0.5 if h in [1,2,3,4] else 1.0)
        mu = -0.00015 if h in [16,17,18] else 0.00008
        ret = np.random.normal(mu, 0.0028 * sv)
        price = max(price*(1+ret), 60000)
        hi  = price*(1+abs(np.random.normal(0,0.0022*sv)))
        lo  = price*(1-abs(np.random.normal(0,0.0022*sv)))
        op  = price*(1+np.random.normal(0,0.0008))
        vol = max(abs(np.random.normal(1200,450))*sv, 80)
        bsk = 0.65 if h in [8,9,10] else (0.36 if h in [17,18,19] else 0.50)
        tb  = vol*np.clip(np.random.beta(bsk*7,(1-bsk)*7),0.05,0.95)
        # Inject occasional big trader candles
        if np.random.random() < 0.03:
            vol *= np.random.uniform(4, 9)
            tb  *= np.random.uniform(3, 8)
        rows.append({
            "open_time":dt,"open":op,"high":hi,"low":lo,"close":price,
            "volume":vol,"taker_buy_vol":tb,"trades":int(vol/0.05)
        })
    return pd.DataFrame(rows)

def synthetic_funding(dates):
    times, rates, prev = [], [], 0.0001
    for dt in dates:
        if dt.hour in [0,8,16]:
            prev = np.clip(prev*0.7+np.random.normal(0.0001,0.00025),-0.002,0.002)
            times.append(dt); rates.append(prev)
    return pd.DataFrame({"fundingTime":times,"fundingRate":rates})

def load_data():
    """Load all timeframes. Falls back to synthetic if no network."""
    print("\n▶  DATA ENGINE — Connecting to Binance Futures...")
    live = False
    dfs = {}
    configs = [
        ("1m",  LOOKBACK_1M),
        ("5m",  LOOKBACK_5M),
        ("1h",  LOOKBACK_1H),
        ("4h",  500),
    ]
    for tf, lim in configs:
        try:
            df = fetch_multi_batch(SYMBOL, tf, lim)
            if not df.empty:
                dfs[tf] = df
                live = True
                print(f"   {tf:>3} ✓  {len(df):>5} candles  "
                      f"{df['open_time'].min().date()} → {df['open_time'].max().date()}")
        except Exception as e:
            pass

    funding = pd.DataFrame()
    if live:
        try:
            funding = fetch_funding(SYMBOL, 500)
            print(f"   Funding ✓  {len(funding)} records")
        except: pass
    else:
        print("   No network — synthesizing institutional-grade BTC data")
        print("   (67,000–68,500 range matching your exact chart context)")
        dfs = {
            "1m":  synthetic_klines(3000, 1,  67200, seed=1),
            "5m":  synthetic_klines(2000, 5,  67000, seed=2),
            "1h":  synthetic_klines(1000, 60, 66800, seed=3),
            "4h":  synthetic_klines(500,  240,65000, seed=4),
        }
        funding = synthetic_funding(dfs["1h"]["open_time"])
        for tf, df in dfs.items():
            print(f"   {tf:>3}    {len(df):>5} candles (synthetic)")

    total = sum(len(v) for v in dfs.values())
    print(f"   Total: {total:,} candles  │  Live: {'YES' if live else 'NO (synthetic)'}")
    return dfs, funding, live


# ══════════════════════════════════════════════════════════════════
#  FEATURE BASE (shared across modules)
# ══════════════════════════════════════════════════════════════════
def build_base(df):
    d = df.copy()
    d["body"]       = d["close"] - d["open"]
    d["body_pct"]   = d["body"] / d["open"] * 100
    d["is_bull"]    = d["body"] > 0
    d["range"]      = d["high"] - d["low"]
    d["range_pct"]  = d["range"] / d["open"] * 100
    d["wick_top"]   = d["high"] - d[["open","close"]].max(axis=1)
    d["wick_bot"]   = d[["open","close"]].min(axis=1) - d["low"]
    d["sell_vol"]   = d["volume"] - d["taker_buy_vol"]
    d["delta"]      = d["taker_buy_vol"] - d["sell_vol"]
    d["delta_pct"]  = d["delta"] / d["volume"].replace(0,np.nan)
    # ATR
    hl  = d["high"] - d["low"]
    hpc = (d["high"] - d["close"].shift(1)).abs()
    lpc = (d["low"]  - d["close"].shift(1)).abs()
    d["atr"] = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    # Volume z-score
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) \
               / d["volume"].rolling(50).std()
    d["hour"]  = d["open_time"].dt.hour
    d["dow"]   = d["open_time"].dt.day_name()
    def sess(h):
        if 0<=h<8:    return "Asia"
        elif 8<=h<13: return "London"
        elif 13<=h<20: return "NY"
        else:          return "Late"
    d["session"] = d["hour"].apply(sess)
    return d.dropna().reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  MODULE 02 │ CVD PRO
# ══════════════════════════════════════════════════════════════════
def cvd_pro(df):
    d = df.copy()
    # Cumulative delta
    d["cvd"]         = d["delta"].cumsum()
    d["cvd_roll20"]  = d["delta"].rolling(20).sum()
    d["cvd_slope3"]  = d["cvd_roll20"].diff(3)
    d["price_slope3"]= d["close"].diff(3) / d["close"].shift(3) * 100
    # Delta exhaustion: delta in direction of candle but very small
    d["delta_exhaust"]= (
        ((d["is_bull"]) & (d["delta_pct"] < 0.05)) |
        ((~d["is_bull"]) & (d["delta_pct"] > -0.05))
    )
    # CVD divergence score
    d["div_bull"] = (d["price_slope3"] < -0.15) & (d["cvd_slope3"] > 0)
    d["div_bear"] = (d["price_slope3"] >  0.15) & (d["cvd_slope3"] < 0)
    d["div_score"]= d["div_bull"].astype(int) - d["div_bear"].astype(int)
    # Absorption: big volume but tiny body (market absorbed)
    d["absorption"]= (d["vol_z"] > 1.5) & (d["body_pct"].abs() < 0.1)

    last  = d.iloc[-1]
    prev5 = d.iloc[-6:-1]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 02 │ CVD PRO                                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Current CVD (roll-20):   {last['cvd_roll20']:>+12.1f}")
    print(f"  CVD slope (3-bar):       {last['cvd_slope3']:>+12.1f}  "
          f"{'↑ buying pressure' if last['cvd_slope3']>0 else '↓ selling pressure'}")
    print(f"  Price slope (3-bar):     {last['price_slope3']:>+12.3f}%")
    bull_div = d["div_bull"].sum()
    bear_div = d["div_bear"].sum()
    absorb   = d["absorption"].sum()
    exhaust  = d["delta_exhaust"].sum()
    print(f"  Bullish CVD divergences: {bull_div:>5}  (price ↓ but CVD ↑)")
    print(f"  Bearish CVD divergences: {bear_div:>5}  (price ↑ but CVD ↓)")
    print(f"  Absorption candles:      {absorb:>5}  (big vol, small body)")
    print(f"  Delta exhaustion:        {exhaust:>5}  (candle vs delta mismatch)")
    if last["div_bull"]:
        print(f"\n  ⚡ ACTIVE SIGNAL: BULLISH CVD DIVERGENCE  ← long bias")
    elif last["div_bear"]:
        print(f"\n  ⚡ ACTIVE SIGNAL: BEARISH CVD DIVERGENCE  ← short bias")
    elif last["absorption"]:
        print(f"\n  ⚡ ACTIVE SIGNAL: ABSORPTION at current price")

    # Recent divergences
    recent_div = d[d["div_bull"] | d["div_bear"]].tail(5)
    if not recent_div.empty:
        print("\n  Recent divergences:")
        for _, row in recent_div.iterrows():
            kind = "BULL" if row["div_bull"] else "BEAR"
            print(f"    {kind} div @ ${row['close']:,.1f}  "
                  f"price_slope={row['price_slope3']:+.3f}%  "
                  f"cvd_slope={row['cvd_slope3']:+.1f}")

    return d


# ══════════════════════════════════════════════════════════════════
#  MODULE 03 │ BIG TRADERS DETECTOR
# ══════════════════════════════════════════════════════════════════
def big_traders(df):
    d = df.copy()
    # Detect candles where volume is anomalously large = institutional/whale
    d["is_big"]     = d["vol_z"] > BIG_TRADE_X
    d["is_moderate"]= (d["vol_z"] > 2.5) & (d["vol_z"] <= BIG_TRADE_X)

    # Iceberg proxy: many trades but small avg trade size
    avg_trade = d["volume"] / d["trades"].replace(0, np.nan)
    global_avg = avg_trade.median()
    d["avg_trade_size"] = avg_trade
    # Iceberg: high volume, high trade count, but small avg size = algo breaking up orders
    d["iceberg_prob"] = (
        (d["vol_z"] > 1.5) &
        (d["trades"] > d["trades"].rolling(50).mean() * 1.5) &
        (avg_trade < global_avg * 0.7)
    )
    # Spoofing proxy: delta completely opposite to direction
    d["spoof_suspect"] = (
        (d["is_bull"] & (d["delta_pct"] < -0.3)) |
        (~d["is_bull"] & (d["delta_pct"] > 0.3))
    )
    # Accumulation: repeated moderate buys at same price level (rolling 10)
    d["accum_signal"] = (
        (~d["is_bull"]) &        # price going down or flat
        (d["delta_pct"] > 0.15) & # but buyers dominating
        (d["vol_z"] > 0.5)
    )
    d["distrib_signal"] = (
        (d["is_bull"]) &
        (d["delta_pct"] < -0.15) &
        (d["vol_z"] > 0.5)
    )

    big_df   = d[d["is_big"]].tail(10)
    iberg_df = d[d["iceberg_prob"]].tail(8)
    accum_df = d[d["accum_signal"]].tail(8)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 03 │ BIG TRADERS DETECTOR                               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Whale candles (vol_z>{BIG_TRADE_X}):     {d['is_big'].sum():>5}")
    print(f"  Large candles (vol_z>2.5):     {d['is_moderate'].sum():>5}")
    print(f"  Iceberg suspected:             {d['iceberg_prob'].sum():>5}")
    print(f"  Spoof suspected:               {d['spoof_suspect'].sum():>5}")
    print(f"  Accumulation signals:          {d['accum_signal'].sum():>5}")
    print(f"  Distribution signals:          {d['distrib_signal'].sum():>5}")

    if not big_df.empty:
        print(f"\n  Recent WHALE candles (vol_z>{BIG_TRADE_X}):")
        for _, r in big_df.iterrows():
            side = "BUY " if r["is_bull"] else "SELL"
            print(f"    {side} @ ${r['close']:>9,.1f}  "
                  f"vol_z={r['vol_z']:>5.1f}  delta={r['delta_pct']:>+.2f}  "
                  f"{r['open_time'].strftime('%m/%d %H:%M')}")

    if not iberg_df.empty:
        print(f"\n  Iceberg order suspicions:")
        for _, r in iberg_df.iterrows():
            print(f"    @ ${r['close']:>9,.1f}  avg_trade={r['avg_trade_size']:.3f} BTC  "
                  f"{r['open_time'].strftime('%m/%d %H:%M')}")

    if not accum_df.empty:
        print(f"\n  Accumulation zones (price falls, buyers absorbing):")
        for _, r in accum_df.iterrows():
            print(f"    @ ${r['close']:>9,.1f}  delta={r['delta_pct']:>+.2f}  "
                  f"vol_z={r['vol_z']:.1f}  {r['open_time'].strftime('%m/%d %H:%M')}")

    return d


# ══════════════════════════════════════════════════════════════════
#  MODULE 04 │ ORDER FLOW ENGINE
# ══════════════════════════════════════════════════════════════════
def order_flow(df):
    d = df.copy()
    # Stacked imbalances: consecutive candles all same delta direction
    d["delta_pos"] = d["delta_pct"] > 0.1
    d["stacked_buy"]  = d["delta_pos"].rolling(3).sum() == 3
    d["stacked_sell"] = (~d["delta_pos"]).rolling(3).sum() == 3

    # Bid absorption: price falls into bids but closes up = buyers absorbed selling
    d["bid_absorb"] = (
        (d["low"] < d["open"]) &
        (d["close"] > d["open"] * 0.999) &
        (d["delta_pct"] > 0.1) &
        (d["vol_z"] > 1.0)
    )
    # Ask absorption: price rises into asks but closes down = sellers absorbed buying
    d["ask_absorb"] = (
        (d["high"] > d["open"]) &
        (d["close"] < d["open"] * 1.001) &
        (d["delta_pct"] < -0.1) &
        (d["vol_z"] > 1.0)
    )
    # Trapped traders: strong candle followed by complete reversal
    d["trapped_longs"]  = (d["body_pct"].shift(1) > 0.25) & (d["close"] < d["open"].shift(1))
    d["trapped_shorts"] = (d["body_pct"].shift(1) < -0.25) & (d["close"] > d["open"].shift(1))

    # Exhaustion: huge delta in direction but price barely moved
    d["buy_exhaust"]  = (d["delta_pct"] > 0.3) & (d["body_pct"] < 0.05)
    d["sell_exhaust"] = (d["delta_pct"] < -0.3) & (d["body_pct"] > -0.05)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 04 │ ORDER FLOW ENGINE                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Stacked buy imbalances:   {d['stacked_buy'].sum():>5}  (3+ bars all buying)")
    print(f"  Stacked sell imbalances:  {d['stacked_sell'].sum():>5}  (3+ bars all selling)")
    print(f"  Bid absorption:           {d['bid_absorb'].sum():>5}  (sellers absorbed)")
    print(f"  Ask absorption:           {d['ask_absorb'].sum():>5}  (buyers absorbed)")
    print(f"  Trapped longs:            {d['trapped_longs'].sum():>5}  (false breakout up)")
    print(f"  Trapped shorts:           {d['trapped_shorts'].sum():>5}  (false breakout down)")
    print(f"  Buyer exhaustion:         {d['buy_exhaust'].sum():>5}  (delta>30% tiny move)")
    print(f"  Seller exhaustion:        {d['sell_exhaust'].sum():>5}  (delta<-30% tiny move)")

    # Current state
    last = d.iloc[-1]
    signals = []
    if last["stacked_buy"]:   signals.append("⚡ STACKED BUY IMBALANCE active")
    if last["stacked_sell"]:  signals.append("⚡ STACKED SELL IMBALANCE active")
    if last["bid_absorb"]:    signals.append("⚡ BID ABSORPTION — buyers defended level")
    if last["ask_absorb"]:    signals.append("⚡ ASK ABSORPTION — sellers defended level")
    if last["buy_exhaust"]:   signals.append("⚠  BUY EXHAUSTION — momentum fading")
    if last["sell_exhaust"]:  signals.append("⚠  SELL EXHAUSTION — selling fading")
    if last["trapped_longs"]: signals.append("⚠  TRAPPED LONGS being squeezed")
    if last["trapped_shorts"]:signals.append("⚠  TRAPPED SHORTS being squeezed")

    if signals:
        print(f"\n  ─── CURRENT ORDER FLOW SIGNALS ───")
        for s in signals: print(f"    {s}")
    else:
        print(f"\n  Current: No strong order flow signal")

    # Recent absorption events
    absorb_events = d[d["bid_absorb"] | d["ask_absorb"]].tail(5)
    if not absorb_events.empty:
        print(f"\n  Recent absorption events:")
        for _, r in absorb_events.iterrows():
            kind = "BID absorb" if r["bid_absorb"] else "ASK absorb"
            print(f"    {kind} @ ${r['close']:>9,.1f}  "
                  f"delta={r['delta_pct']:>+.2f}  vol_z={r['vol_z']:.1f}  "
                  f"{r['open_time'].strftime('%m/%d %H:%M')}")
    return d


# ══════════════════════════════════════════════════════════════════
#  MODULE 05 │ UNFINISHED BUSINESS
# ══════════════════════════════════════════════════════════════════
def unfinished_business(df):
    d = df.copy()
    current_price = d["close"].iloc[-1]

    # ── Unfilled gaps
    gaps = []
    for i in range(1, len(d)):
        ph, pl = d["high"].iloc[i-1], d["low"].iloc[i-1]
        co = d["open"].iloc[i]
        if co > ph * 1.0008:
            future = d["low"].iloc[i:]
            filled = future.min() <= ph
            gaps.append({"dir":"↑ BULL","level":round(ph,1),
                         "gap_%":round((co-ph)/ph*100,3),
                         "filled":filled,"dist_%":round((current_price-ph)/ph*100,2),
                         "time":d["open_time"].iloc[i].strftime("%m/%d %H:%M")})
        elif co < pl * 0.9992:
            future = d["high"].iloc[i:]
            filled = future.max() >= pl
            gaps.append({"dir":"↓ BEAR","level":round(pl,1),
                         "gap_%":round((pl-co)/pl*100,3),
                         "filled":filled,"dist_%":round((pl-current_price)/pl*100,2),
                         "time":d["open_time"].iloc[i].strftime("%m/%d %H:%M")})

    gap_df = pd.DataFrame(gaps)
    unfilled = gap_df[~gap_df["filled"]].copy() if not gap_df.empty else pd.DataFrame()
    if not unfilled.empty:
        unfilled = unfilled.sort_values("dist_%").head(10)

    # ── Naked high-volume nodes (top 3% vol, never revisited)
    thresh = d["volume"].quantile(0.97)
    nodes = []
    for i, row in d[d["volume"] >= thresh].iterrows():
        level = (row["high"] + row["low"]) / 2
        fut   = d[d["open_time"] > row["open_time"]]
        if fut.empty: continue
        if not ((fut["low"] <= level) & (fut["high"] >= level)).any():
            nodes.append({
                "level":    round(level, 1),
                "vol_z":    round(row["vol_z"], 1),
                "dir":      "BULL" if row["is_bull"] else "BEAR",
                "dist_$":   round(current_price - level, 1),
                "time":     row["open_time"].strftime("%m/%d %H:%M"),
            })
    node_df = pd.DataFrame(nodes).sort_values("dist_$", key=abs).head(8) \
              if nodes else pd.DataFrame()

    # ── Failed auctions (spike + reversal > 1.8x ATR)
    failed = []
    atr_med = d["atr"].median()
    for i in range(5, len(d)-3):
        atr = d["atr"].iloc[i]
        if atr <= 0: continue
        # Spike high
        if d["high"].iloc[i] == d["high"].iloc[i-3:i+3].max():
            wick = d["wick_top"].iloc[i] / atr
            if wick > 1.8:
                failed.append({"type":"SPIKE HI","level":round(d["high"].iloc[i],1),
                               "wick_atr":round(wick,2),
                               "dist_$":round(current_price-d["high"].iloc[i],1),
                               "time":d["open_time"].iloc[i].strftime("%m/%d %H:%M")})
        # Spike low
        if d["low"].iloc[i] == d["low"].iloc[i-3:i+3].min():
            wick = d["wick_bot"].iloc[i] / atr
            if wick > 1.8:
                failed.append({"type":"SPIKE LO","level":round(d["low"].iloc[i],1),
                               "wick_atr":round(wick,2),
                               "dist_$":round(current_price-d["low"].iloc[i],1),
                               "time":d["open_time"].iloc[i].strftime("%m/%d %H:%M")})
    fail_df = pd.DataFrame(failed).sort_values("dist_$", key=abs).head(8) \
              if failed else pd.DataFrame()

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 05 │ UNFINISHED BUSINESS                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Current price: ${current_price:,.1f}")

    print(f"\n  ── Unfilled Gaps ({len(unfilled)} nearest):")
    if not unfilled.empty:
        print(unfilled.to_string(index=False))
    else:
        print("    None found")

    print(f"\n  ── Naked Volume Nodes ({len(node_df)} nearest):")
    if not node_df.empty:
        print(node_df.to_string(index=False))
    else:
        print("    None found")

    print(f"\n  ── Failed Auctions ({len(fail_df)}):")
    if not fail_df.empty:
        print(fail_df.to_string(index=False))
    else:
        print("    None found")

    return unfilled, node_df, fail_df


# ══════════════════════════════════════════════════════════════════
#  MODULE 06 │ VWAP / TWAP ENGINE
# ══════════════════════════════════════════════════════════════════
def vwap_twap(df):
    d = df.copy()
    tp = (d["high"] + d["low"] + d["close"]) / 3

    # Rolling VWAP (20, 50, 200)
    for n in [20, 50, 200]:
        if len(d) >= n:
            d[f"vwap{n}"] = (tp * d["volume"]).rolling(n).sum() \
                           / d["volume"].rolling(n).sum()
            d[f"vwap{n}_dev"] = (d["close"] - d[f"vwap{n}"]) / d[f"vwap{n}"] * 100

    # Session VWAP (reset each trading day)
    d["date"] = d["open_time"].dt.date
    d["sess_tp_vol"] = tp * d["volume"]
    d["sess_vwap"] = d.groupby("date")["sess_tp_vol"].cumsum() \
                   / d.groupby("date")["volume"].cumsum()
    d["sess_dev"] = (d["close"] - d["sess_vwap"]) / d["sess_vwap"] * 100

    # TWAP (20-bar simple mean of typical price)
    d["twap20"] = tp.rolling(20).mean()
    d["twap_dev"] = (d["close"] - d["twap20"]) / d["twap20"] * 100

    # VWAP bands (1 and 2 std dev)
    vwap_vol = (d["volume"] * (tp - d["vwap20"])**2).rolling(20).sum()
    vol_sum  = d["volume"].rolling(20).sum()
    d["vwap_std"] = np.sqrt(vwap_vol / vol_sum.replace(0, np.nan))
    d["vwap_u1"]  = d["vwap20"] + d["vwap_std"]
    d["vwap_l1"]  = d["vwap20"] - d["vwap_std"]
    d["vwap_u2"]  = d["vwap20"] + 2*d["vwap_std"]
    d["vwap_l2"]  = d["vwap20"] - 2*d["vwap_std"]

    last = d.iloc[-1]
    price = last["close"]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 06 │ VWAP / TWAP ENGINE                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Current price:    ${price:>11,.1f}")
    print(f"  Session VWAP:     ${last.get('sess_vwap', np.nan):>11,.1f}  "
          f"dev={last.get('sess_dev',0):>+.3f}%")
    print(f"  VWAP(20):         ${last.get('vwap20', np.nan):>11,.1f}  "
          f"dev={last.get('vwap20_dev',0):>+.3f}%")
    print(f"  VWAP(50):         ${last.get('vwap50', np.nan):>11,.1f}  "
          f"dev={last.get('vwap50_dev',0):>+.3f}%")
    if "vwap200" in last.index:
        print(f"  VWAP(200):        ${last.get('vwap200', np.nan):>11,.1f}  "
              f"dev={last.get('vwap200_dev',0):>+.3f}%")
    print(f"  TWAP(20):         ${last.get('twap20', np.nan):>11,.1f}  "
          f"dev={last.get('twap_dev',0):>+.3f}%")
    print(f"  VWAP +1σ:         ${last.get('vwap_u1', np.nan):>11,.1f}")
    print(f"  VWAP -1σ:         ${last.get('vwap_l1', np.nan):>11,.1f}")
    print(f"  VWAP +2σ:         ${last.get('vwap_u2', np.nan):>11,.1f}")
    print(f"  VWAP -2σ:         ${last.get('vwap_l2', np.nan):>11,.1f}")

    # VWAP position
    sess_vwap = last.get("sess_vwap", price)
    if price > sess_vwap * 1.003:
        print(f"\n  Position: EXTENDED ABOVE session VWAP (+{(price/sess_vwap-1)*100:.2f}%)")
        print(f"  → Bias: mean-reversion risk. Short opportunities at +2σ band.")
    elif price < sess_vwap * 0.997:
        print(f"\n  Position: EXTENDED BELOW session VWAP ({(price/sess_vwap-1)*100:.2f}%)")
        print(f"  → Bias: mean-reversion long potential. Buy at -2σ with delta confirm.")
    else:
        print(f"\n  Position: Near session VWAP → Price discovery zone, no strong bias.")

    return d


# ══════════════════════════════════════════════════════════════════
#  MODULE 07 │ MARKET PROFILE
# ══════════════════════════════════════════════════════════════════
def market_profile(df, tick_size=10.0):
    d = df.copy()
    price = d["close"].iloc[-1]

    # Build volume profile: bucket volume by price level
    lo  = d["low"].min()
    hi  = d["high"].max()
    buckets = np.arange(np.floor(lo/tick_size)*tick_size,
                        np.ceil(hi/tick_size)*tick_size + tick_size,
                        tick_size)

    vol_profile = defaultdict(float)
    buy_profile = defaultdict(float)
    sell_profile= defaultdict(float)

    for _, row in d.iterrows():
        candle_buckets = buckets[(buckets >= row["low"]) & (buckets <= row["high"])]
        if len(candle_buckets) == 0: continue
        vol_per = row["volume"] / len(candle_buckets)
        buy_per = row["taker_buy_vol"] / len(candle_buckets)
        sell_per= row["sell_vol"] / len(candle_buckets)
        for b in candle_buckets:
            vol_profile[b]  += vol_per
            buy_profile[b]  += buy_per
            sell_profile[b] += sell_per

    profile_df = pd.DataFrame({
        "price":  list(vol_profile.keys()),
        "volume": list(vol_profile.values()),
        "buys":   [buy_profile[k]  for k in vol_profile],
        "sells":  [sell_profile[k] for k in vol_profile],
    }).sort_values("price")

    total_vol = profile_df["volume"].sum()
    profile_df["vol_pct"] = profile_df["volume"] / total_vol

    # POC = price of control (highest volume)
    poc = profile_df.loc[profile_df["volume"].idxmax(), "price"]

    # Value Area (70% of volume around POC)
    poc_idx  = profile_df["volume"].idxmax()
    cum_vol  = 0
    va_upper = poc
    va_lower = poc
    va_rows  = [poc_idx]
    while cum_vol / total_vol < 0.70:
        upper_i = max(va_rows) + 1
        lower_i = min(va_rows) - 1
        up_vol  = profile_df.loc[upper_i, "volume"] if upper_i in profile_df.index else 0
        dn_vol  = profile_df.loc[lower_i, "volume"] if lower_i in profile_df.index else 0
        if up_vol >= dn_vol and upper_i in profile_df.index:
            va_rows.append(upper_i)
            cum_vol += up_vol
        elif lower_i in profile_df.index:
            va_rows.append(lower_i)
            cum_vol += dn_vol
        else:
            break
    va_df   = profile_df.loc[va_rows]
    vah     = va_df["price"].max()
    val     = va_df["price"].min()

    # Distribution shape
    top_half  = profile_df[profile_df["price"] > poc]["volume"].sum()
    bot_half  = profile_df[profile_df["price"] < poc]["volume"].sum()
    if   top_half > bot_half * 1.3: shape = "P-shape (buying tail up)"
    elif bot_half > top_half * 1.3: shape = "b-shape (selling tail down)"
    elif (total_vol*0.6) < profile_df.nlargest(5,"volume")["volume"].sum(): shape = "D-shape (balanced)"
    else: shape = "Elongated (trending/directional)"

    # Thin zones (gaps in profile = price will traverse fast)
    thin = profile_df[profile_df["vol_pct"] < 0.002].sort_values("price")

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 07 │ MARKET PROFILE                                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  POC (Point of Control): ${poc:>10,.1f}")
    print(f"  VAH (Value Area High):  ${vah:>10,.1f}")
    print(f"  VAL (Value Area Low):   ${val:>10,.1f}")
    print(f"  Value Area Width:       ${vah-val:>10,.1f}")
    print(f"  Distribution shape:     {shape}")
    print(f"  Current price vs POC:   {'+' if price>=poc else ''}{price-poc:,.1f}  "
          f"{'(above POC)' if price>poc else '(below POC)'}")

    # Price position relative to value
    if price > vah:
        print(f"\n  PRICE ABOVE VALUE AREA  →  Acceptance above = bullish")
        print(f"  OR Rejection and return to VA = fade opportunity")
    elif price < val:
        print(f"\n  PRICE BELOW VALUE AREA  →  Acceptance below = bearish")
        print(f"  OR Rejection and return to VA = buy opportunity at VAL")
    else:
        print(f"\n  PRICE INSIDE VALUE AREA  →  Mean-reversion regime")
        print(f"  Trade: short near VAH, long near VAL, target POC")

    # Top 5 volume nodes
    print(f"\n  Top 5 volume nodes (magnetic levels):")
    for _, r in profile_df.nlargest(5,"volume").sort_values("price",ascending=False).iterrows():
        bar = "█" * int(r["vol_pct"] * 200)
        marker = " ← CURRENT" if abs(r["price"]-price) < tick_size*2 else ""
        print(f"    ${r['price']:>9,.1f}  {bar:<20}  {r['vol_pct']*100:>5.2f}%{marker}")

    if len(thin) > 0:
        thin_near = thin[((thin["price"]-price).abs() < (hi-lo)*0.3)]
        if not thin_near.empty:
            print(f"\n  Thin zones (price will move FAST through these):")
            for _, r in thin_near.head(5).iterrows():
                print(f"    ${r['price']:>9,.1f}  (only {r['vol_pct']*100:.3f}% of volume)")

    return poc, vah, val, profile_df


# ══════════════════════════════════════════════════════════════════
#  MODULE 08 │ FOOTPRINT ANALYSIS
# ══════════════════════════════════════════════════════════════════
def footprint_analysis(df, n_candles=10, tick_size=10.0):
    """Simulate footprint chart: bid/ask volume at each price level per candle."""
    d = df.tail(n_candles).copy()

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 08 │ FOOTPRINT ANALYSIS                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Showing last {n_candles} candles | tick_size=${tick_size}")
    print(f"  Format:  PRICE │ BUY_VOL × SELL_VOL │ DELTA │ NOTE")
    print(f"  {'─'*66}")

    for _, row in d.iterrows():
        candle_range = row["high"] - row["low"]
        if candle_range < tick_size: continue
        levels = np.arange(
            np.floor(row["low"]/tick_size)*tick_size,
            np.ceil(row["high"]/tick_size)*tick_size,
            tick_size
        )
        if len(levels) == 0: continue
        # Distribute volume: more volume near close (completed trades)
        weights = np.ones(len(levels))
        close_idx = np.argmin(np.abs(levels - row["close"]))
        weights[close_idx] *= 2.5
        weights = weights / weights.sum()
        buy_vols  = row["taker_buy_vol"] * weights
        sell_vols = row["sell_vol"] * weights

        # Print candle header
        dir_sym = "▲" if row["is_bull"] else "▼"
        print(f"\n  {dir_sym} {row['open_time'].strftime('%H:%M')}  "
              f"O:{row['open']:,.0f} H:{row['high']:,.0f} "
              f"L:{row['low']:,.0f} C:{row['close']:,.0f}  "
              f"Δ={row['delta_pct']:>+.2f}  vol_z={row['vol_z']:.1f}")

        # Print levels (top 6 most significant)
        level_data = list(zip(levels, buy_vols, sell_vols))
        level_data.sort(key=lambda x: x[1]+x[2], reverse=True)
        for lvl, bv, sv in level_data[:6]:
            delta = bv - sv
            note = ""
            if bv > sv * IMBALANCE_THR: note = "  ← BID STACK"
            if sv > bv * IMBALANCE_THR: note = "  ← ASK STACK"
            if abs(lvl - row["close"]) < tick_size/2: note += " [CLOSE]"
            bar_b = "▶" * min(int(bv/row["volume"]*40), 20)
            bar_s = "◀" * min(int(sv/row["volume"]*40), 20)
            print(f"    ${lvl:>9,.0f} │ {bar_b:<20} {bv:>7.1f}B  "
                  f"{sv:>7.1f}S {bar_s:<20} │ Δ{delta:>+8.1f}{note}")


# ══════════════════════════════════════════════════════════════════
#  MODULE 09 │ TPO ANALYSIS
# ══════════════════════════════════════════════════════════════════
def tpo_analysis(df, tick_size=25.0, period_minutes=30):
    """Build TPO (Time Price Opportunity) chart."""
    d = df.copy()
    d["tpo_period"] = d["open_time"].dt.floor(f"{period_minutes}min")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    all_prices = set()
    period_map = {}
    for i, (period, grp) in enumerate(d.groupby("tpo_period")):
        letter = letters[i % len(letters)]
        prices_in_period = set()
        for _, row in grp.iterrows():
            lvls = np.arange(
                np.floor(row["low"]/tick_size)*tick_size,
                np.ceil(row["high"]/tick_size)*tick_size,
                tick_size
            )
            for p in lvls:
                prices_in_period.add(p)
                all_prices.add(p)
        period_map[letter] = prices_in_period

    # Count TPO hits per price
    tpo_count = defaultdict(int)
    for letter, prices in period_map.items():
        for p in prices:
            tpo_count[p] += 1

    poc_price = max(tpo_count, key=tpo_count.get) if tpo_count else 0

    # Initial Balance (first 2 periods = first hour)
    ib_letters = list(period_map.keys())[:2]
    ib_prices  = set()
    for l in ib_letters:
        ib_prices.update(period_map.get(l, set()))
    ib_high = max(ib_prices) if ib_prices else 0
    ib_low  = min(ib_prices) if ib_prices else 0

    # Extensions beyond IB
    ext_up = [p for p in all_prices if p > ib_high]
    ext_dn = [p for p in all_prices if p < ib_low]

    # Single prints (TPO count = 1 = thin zone)
    single_prints = sorted([p for p,c in tpo_count.items() if c == 1])
    current_price = d["close"].iloc[-1]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 09 │ TPO ANALYSIS                                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Tick size:     ${tick_size}  │  Period: {period_minutes}min")
    print(f"  Periods built: {len(period_map)}")
    print(f"  POC (TPO):     ${poc_price:>10,.0f}")
    print(f"  IB High:       ${ib_high:>10,.0f}")
    print(f"  IB Low:        ${ib_low:>10,.0f}")
    print(f"  IB Range:      ${ib_high-ib_low:>10,.0f}")
    print(f"  Extensions UP: {len(ext_up)} ticks  → max ${max(ext_up):.0f}" if ext_up else "  Extensions UP: None")
    print(f"  Extensions DN: {len(ext_dn)} ticks  → min ${min(ext_dn):.0f}" if ext_dn else "  Extensions DN: None")
    print(f"  Single prints: {len(single_prints)} levels  (thin = fast travel zones)")

    # Print TPO chart (last 20 price levels around current price)
    sorted_prices = sorted(all_prices, reverse=True)
    nearby = [p for p in sorted_prices if abs(p-current_price) < tick_size*20][:30]
    print(f"\n  TPO Chart (near current price ${current_price:,.0f}):")
    print(f"  {'PRICE':>10}  TPO_LETTERS                    COUNT  NOTE")
    for p in nearby:
        tpo_letters = "".join(l for l,ps in period_map.items() if p in ps)
        count = tpo_count.get(p, 0)
        note = ""
        if p == poc_price: note = " ← TPO POC"
        if abs(p - ib_high) < tick_size/2: note = " ← IB HIGH"
        if abs(p - ib_low)  < tick_size/2: note = " ← IB LOW"
        if count == 1: note += " [SINGLE PRINT]"
        cursor = " ◄" if abs(p - current_price) < tick_size else ""
        print(f"  ${p:>9,.0f}  {tpo_letters:<30} {count:>3}{note}{cursor}")

    return poc_price, ib_high, ib_low, single_prints


# ══════════════════════════════════════════════════════════════════
#  MODULE 10 │ IMBALANCE CHART
# ══════════════════════════════════════════════════════════════════
def imbalance_chart(df, n_candles=20, tick_size=10.0):
    """Detect and map bid/ask imbalances (stacked imbalance zones)."""
    d = df.tail(n_candles).copy()
    price = d["close"].iloc[-1]

    # Build imbalance map
    buy_imbal  = defaultdict(float)  # price levels with aggressive buying
    sell_imbal = defaultdict(float)  # price levels with aggressive selling

    for _, row in d.iterrows():
        levels = np.arange(
            np.floor(row["low"]/tick_size)*tick_size,
            np.ceil(row["high"]/tick_size)*tick_size,
            tick_size
        )
        if len(levels) == 0: continue
        # Weight distribution
        weights = np.ones(len(levels))
        weights = weights / weights.sum()
        bv = row["taker_buy_vol"] * weights
        sv = row["sell_vol"] * weights

        for i, lvl in enumerate(levels):
            ratio = bv[i] / sv[i] if sv[i] > 0.01 else 10
            if ratio >= IMBALANCE_THR:
                buy_imbal[lvl]  += bv[i]
            elif ratio <= 1/IMBALANCE_THR:
                sell_imbal[lvl] += sv[i]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 10 │ IMBALANCE CHART                                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Threshold: buy/sell ratio ≥ {IMBALANCE_THR}x = imbalance")
    print(f"  Levels with BUY  imbalance: {len(buy_imbal)}")
    print(f"  Levels with SELL imbalance: {len(sell_imbal)}")

    # Merge and display
    all_levels = sorted(set(list(buy_imbal.keys()) + list(sell_imbal.keys())), reverse=True)
    near = [l for l in all_levels if abs(l-price) < tick_size*30]

    print(f"\n  Imbalance map (near ${price:,.0f}):")
    print(f"  {'PRICE':>10}  {'BUY IMBAL':>12}  {'SELL IMBAL':>12}  DOMINANT")
    stacked_buy_zones  = []
    stacked_sell_zones = []
    prev_was_buy = None
    stack_start  = None
    stack_count  = 0

    for lvl in sorted(near, reverse=True):
        bv = buy_imbal.get(lvl, 0)
        sv = sell_imbal.get(lvl, 0)
        if bv > 0 and sv == 0:
            dom = "BUY  ▲▲▲"
            is_buy = True
        elif sv > 0 and bv == 0:
            dom = "SELL ▼▼▼"
            is_buy = False
        else:
            dom = "mixed"
            is_buy = None

        # Track stacked zones
        if is_buy == prev_was_buy and is_buy is not None:
            stack_count += 1
        else:
            if stack_count >= 3:
                if prev_was_buy:
                    stacked_buy_zones.append((stack_start, lvl+tick_size))
                else:
                    stacked_sell_zones.append((stack_start, lvl+tick_size))
            stack_count = 1
            stack_start = lvl
        prev_was_buy = is_buy

        cursor = " ◄ PRICE" if abs(lvl-price) < tick_size else ""
        print(f"  ${lvl:>9,.0f}  {bv:>12.1f}  {sv:>12.1f}  {dom}{cursor}")

    if stacked_buy_zones:
        print(f"\n  ⚡ STACKED BUY ZONES (strong support):")
        for lo, hi in stacked_buy_zones:
            print(f"    ${lo:,.0f} – ${hi:,.0f}")
    if stacked_sell_zones:
        print(f"\n  ⚡ STACKED SELL ZONES (strong resistance):")
        for lo, hi in stacked_sell_zones:
            print(f"    ${lo:,.0f} – ${hi:,.0f}")

    return buy_imbal, sell_imbal


# ══════════════════════════════════════════════════════════════════
#  MODULE 11 │ LIQUIDITY MAP
# ══════════════════════════════════════════════════════════════════
def liquidity_map(df, df_htf=None):
    """
    Map where stop orders and liquidations are likely clustered.
    Based on: equal highs/lows, swing points, round numbers, ATR clusters.
    """
    d = df.copy()
    price = d["close"].iloc[-1]
    atr   = d["atr"].iloc[-1]

    # ── Equal highs/lows (stop clusters)
    tolerance = atr * 0.3
    eq_highs, eq_lows = [], []
    for i in range(5, len(d)-2):
        h = d["high"].iloc[i]
        nearby_h = d["high"].iloc[max(0,i-10):i]
        if ((nearby_h - h).abs() < tolerance).any():
            eq_highs.append(round(h, -1))
        l = d["low"].iloc[i]
        nearby_l = d["low"].iloc[max(0,i-10):i]
        if ((nearby_l - l).abs() < tolerance).any():
            eq_lows.append(round(l, -1))

    from collections import Counter
    eq_h_counts = Counter(eq_highs).most_common(5)
    eq_l_counts = Counter(eq_lows).most_common(5)

    # ── Round number magnets (psychological stops)
    round_levels = []
    for mult in [100, 250, 500, 1000]:
        lo_r = np.floor(price/mult) * mult
        hi_r = np.ceil(price/mult)  * mult
        round_levels.extend([lo_r - mult, lo_r, hi_r, hi_r + mult])
    round_levels = sorted(set([r for r in round_levels if abs(r-price) < atr*15]))

    # ── Swing highs/lows (most stops live just beyond these)
    lookback = 10
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(d)-lookback):
        if d["high"].iloc[i] == d["high"].iloc[i-lookback:i+lookback].max():
            swing_highs.append(d["high"].iloc[i])
        if d["low"].iloc[i]  == d["low"].iloc[i-lookback:i+lookback].min():
            swing_lows.append(d["low"].iloc[i])

    # Nearest swing levels
    near_sh = sorted(swing_highs, key=lambda x: abs(x-price))[:5]
    near_sl = sorted(swing_lows,  key=lambda x: abs(x-price))[:5]

    # ── Vol spike clusters (likely liquidation events already happened here)
    liq_events = d[d["vol_z"] > 2.5][["open_time","close","vol_z","delta_pct","is_bull"]]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 11 │ LIQUIDITY MAP                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Current price: ${price:,.1f}  │  ATR: ${atr:.1f}")

    print(f"\n  ── Equal Highs (stop clusters ABOVE) ───")
    for level, count in eq_h_counts:
        dist = level - price
        stars = "★" * min(count, 5)
        print(f"    ${level:>10,.0f}  {stars:<6}  count={count}  dist=+${dist:,.0f}  "
              f"← SHORT STOPS ABOVE")

    print(f"\n  ── Equal Lows (stop clusters BELOW) ───")
    for level, count in eq_l_counts:
        dist = price - level
        stars = "★" * min(count, 5)
        print(f"    ${level:>10,.0f}  {stars:<6}  count={count}  dist=-${dist:,.0f}  "
              f"← LONG STOPS BELOW")

    print(f"\n  ── Round Number Magnets ───")
    for lvl in round_levels:
        dist   = lvl - price
        marker = " ◄ NEAREST" if abs(dist) < atr*2 else ""
        print(f"    ${lvl:>10,.0f}  dist={dist:>+8,.0f}{marker}")

    print(f"\n  ── Swing Highs (stops just above) ───")
    for h in sorted(near_sh, reverse=True):
        if h > price:
            print(f"    ${h:>10,.1f}  above by ${h-price:,.0f}  ← STOPS AT {h+atr*0.5:.0f}")
    print(f"\n  ── Swing Lows (stops just below) ───")
    for l in sorted(near_sl, reverse=True):
        if l < price:
            print(f"    ${l:>10,.1f}  below by ${price-l:,.0f}  ← STOPS AT {l-atr*0.5:.0f}")

    if not liq_events.empty:
        print(f"\n  ── Recent Liquidation Events (vol_z>2.5) ───")
        for _, r in liq_events.tail(6).iterrows():
            side = "LONG LIQ" if not r["is_bull"] else "SHORT LIQ"
            print(f"    {side} @ ${r['close']:>9,.1f}  "
                  f"vol_z={r['vol_z']:.1f}  {r['open_time'].strftime('%m/%d %H:%M')}")

    return near_sh, near_sl, round_levels


# ══════════════════════════════════════════════════════════════════
#  MODULE 12 │ HEDGE FUND LAYER
# ══════════════════════════════════════════════════════════════════
def hedge_fund_analysis(df, poc, vah, val, funding_df):
    """
    Institutional / hedge fund behavioral analysis.
    What would a Renaissance, Citadel, or Two Sigma quant look for?
    """
    d = df.copy()
    price = d["close"].iloc[-1]
    atr   = d["atr"].iloc[-1]

    # ── Accumulation vs Distribution (Wyckoff)
    # Look for: declining price + increasing buy volume = accumulation
    # OR: rising price + increasing sell volume = distribution
    recent = d.tail(30)
    price_trend  = np.polyfit(range(len(recent)), recent["close"], 1)[0]
    volume_trend = np.polyfit(range(len(recent)), recent["volume"], 1)[0]
    buy_vol_trend= np.polyfit(range(len(recent)), recent["taker_buy_vol"], 1)[0]
    sell_vol_trend=np.polyfit(range(len(recent)), recent["sell_vol"], 1)[0]

    if price_trend < -0.5 and buy_vol_trend > 0:
        wyckoff = "ACCUMULATION  (smart money buying the dip)"
        wyck_bias = 1
    elif price_trend > 0.5 and sell_vol_trend > 0:
        wyckoff = "DISTRIBUTION  (smart money selling the rally)"
        wyck_bias = -1
    elif price_trend < 0 and sell_vol_trend < 0:
        wyckoff = "MARKDOWN      (no buyers, price falling)"
        wyck_bias = -2
    elif price_trend > 0 and buy_vol_trend > 0:
        wyckoff = "MARKUP        (buyers in control, trending up)"
        wyck_bias = 2
    else:
        wyckoff = "CONSOLIDATION (no clear institutional footprint)"
        wyck_bias = 0

    # ── Smart money flow (CVD divergence across lookback)
    d["cvd_20"] = d["delta"].rolling(20).sum()
    cvd_now  = d["cvd_20"].iloc[-1]
    cvd_prev = d["cvd_20"].iloc[-20] if len(d) > 20 else cvd_now
    cvd_trend= "Rising (net buying)" if cvd_now > cvd_prev else "Falling (net selling)"

    # ── Funding rate analysis
    fund_signal = "Neutral"
    fund_bias   = 0
    if not funding_df.empty:
        recent_fr  = funding_df["fundingRate"].tail(8)
        avg_fr     = recent_fr.mean()
        if avg_fr > 0.0005:
            fund_signal = f"OVERHEATED LONGS (avg={avg_fr*100:.4f}%) → mean revert DOWN likely"
            fund_bias   = -1
        elif avg_fr < -0.0003:
            fund_signal = f"OVERHEATED SHORTS (avg={avg_fr*100:.4f}%) → short squeeze UP likely"
            fund_bias   = 1
        else:
            fund_signal = f"Neutral (avg={avg_fr*100:.4f}%)"

    # ── Institutional rotation (where is volume concentrating?)
    vol_above_poc = d[d["close"] > poc]["volume"].sum()
    vol_below_poc = d[d["close"] < poc]["volume"].sum()
    vol_rotation  = "Rotating UP (volume above POC)" if vol_above_poc > vol_below_poc \
                    else "Rotating DOWN (volume below POC)"

    # ── Market regime (trending vs. mean-reverting)
    hurst_proxy = recent["close"].rolling(10).std().mean() / \
                  recent["close"].rolling(30).std().mean() if len(recent) >= 30 else 0.5
    if   hurst_proxy > 0.7: regime = "TRENDING      → follow momentum"
    elif hurst_proxy < 0.4: regime = "MEAN-REVERTING → fade extremes"
    else:                   regime = "MIXED          → trade both"

    # ── Hedge fund price targets
    targets_up = [
        vah,
        round(price + atr*2, -1),
        round(price + atr*4, -1),
    ]
    targets_dn = [
        val,
        round(price - atr*2, -1),
        round(price - atr*4, -1),
    ]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 12 │ HEDGE FUND LAYER                                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  ── Wyckoff Market Structure ─────────────────────────────────")
    print(f"    Phase:      {wyckoff}")
    print(f"    Price trend (30-bar): {price_trend:>+.2f} $/candle")
    print(f"    Buy vol trend:        {'RISING ↑' if buy_vol_trend>0 else 'FALLING ↓'}")
    print(f"    Sell vol trend:       {'RISING ↑' if sell_vol_trend>0 else 'FALLING ↓'}")

    print(f"\n  ── Smart Money Flow ─────────────────────────────────────────")
    print(f"    CVD(20):    {cvd_now:>+,.1f}  ({cvd_trend})")
    print(f"    vs 20 bars ago: {cvd_prev:>+,.1f}  → {'NET BUY' if cvd_now>cvd_prev else 'NET SELL'}")
    print(f"    Volume rotation: {vol_rotation}")

    print(f"\n  ── Funding Rate Intelligence ────────────────────────────────")
    print(f"    Signal:  {fund_signal}")

    print(f"\n  ── Market Regime ────────────────────────────────────────────")
    print(f"    Regime:  {regime}")
    print(f"    Hurst proxy: {hurst_proxy:.3f}")

    print(f"\n  ── Institutional Price Targets ──────────────────────────────")
    print(f"    Targets UP:  " + "  |  ".join(f"${t:,.0f}" for t in targets_up))
    print(f"    Targets DN:  " + "  |  ".join(f"${t:,.0f}" for t in targets_dn))

    print(f"\n  ── Hedge Fund Perspective (from pattern library) ────────────")
    print(f"""
    "The best short-term traders I've known trade patterns that repeat
     with a statistical edge. They don't have opinions on Bitcoin.
     They have opinions on order flow, value, and where the crowd is trapped."
     — Institutional trading desk principle

    Current read:
    → Wyckoff bias:  {'+1 BULL' if wyck_bias>0 else ('-1 BEAR' if wyck_bias<0 else 'NEUTRAL')}
    → CVD bias:      {'BULL' if cvd_now>cvd_prev else 'BEAR'}
    → Funding bias:  {'BULL' if fund_bias>0 else ('BEAR' if fund_bias<0 else 'NEUTRAL')}
    → POC position:  {'ABOVE POC (bullish context)' if price>poc else 'BELOW POC (bearish context)'}
    → Value area:    {'ABOVE VA (breakout or fade)' if price>vah else ('BELOW VA (breakdown or buy)' if price<val else 'INSIDE VA (mean revert)')}""")

    return wyck_bias, fund_bias, cvd_now > cvd_prev


# ══════════════════════════════════════════════════════════════════
#  MODULE 13 │ SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════
def signal_engine(df, poc, vah, val, wyck_bias, fund_bias, cvd_net_bull,
                  near_sh, near_sl, unf_gaps):
    d = df.copy()
    last  = d.iloc[-1]
    price = last["close"]
    atr   = last["atr"]

    # ── Build composite score (-10 to +10)
    score = 0
    reasons = []

    # CVD divergence
    if last.get("div_bull", False):
        score += 2; reasons.append("✦ Bullish CVD divergence  +2")
    elif last.get("div_bear", False):
        score -= 2; reasons.append("✦ Bearish CVD divergence  -2")

    # Wyckoff
    score += wyck_bias
    reasons.append(f"✦ Wyckoff phase  {wyck_bias:+d}")

    # Funding
    score += fund_bias
    reasons.append(f"✦ Funding regime  {fund_bias:+d}")

    # CVD net
    if cvd_net_bull: score += 1; reasons.append("✦ CVD net buying (20-bar)  +1")
    else:            score -= 1; reasons.append("✦ CVD net selling (20-bar)  -1")

    # Delta
    if last["delta_pct"] > 0.2:
        score += 1; reasons.append(f"✦ Strong buy delta ({last['delta_pct']:+.2f})  +1")
    elif last["delta_pct"] < -0.2:
        score -= 1; reasons.append(f"✦ Strong sell delta ({last['delta_pct']:+.2f})  -1")

    # VWAP
    if last.get("vwap20_dev", 0) > 0.3:
        score -= 1; reasons.append("✦ Extended above VWAP  -1")
    elif last.get("vwap20_dev", 0) < -0.3:
        score += 1; reasons.append("✦ Extended below VWAP  +1")

    # Value area
    if price > vah:
        score += 1; reasons.append("✦ Above Value Area  +1")
    elif price < val:
        score -= 1; reasons.append("✦ Below Value Area  -1")

    # Vol spike
    if last["vol_z"] > 2.5:
        v = 1 if last["is_bull"] else -1
        score += v; reasons.append(f"✦ Vol spike (z={last['vol_z']:.1f})  {v:+d}")

    # Absorption
    if last.get("absorption", False):
        reasons.append("✦ Absorption candle (monitor direction)")

    # Wick rejection
    if last["wick_top"] > atr * 0.3:
        score -= 1; reasons.append(f"✦ Top wick rejection  -1")
    if last["wick_bot"] > atr * 0.3:
        score += 1; reasons.append(f"✦ Bottom wick support  +1")

    # Trapped traders
    if last.get("trapped_longs", False):
        score -= 1; reasons.append("✦ Trapped longs detected  -1")
    if last.get("trapped_shorts", False):
        score += 1; reasons.append("✦ Trapped shorts detected  +1")

    # Session bonus
    if last["session"] in ["London", "NY"]:
        reasons.append(f"✦ Active session ({last['session']})  +0 (but higher quality signals)")

    # ── Final bias
    score = np.clip(score, -10, 10)
    confidence = abs(score) / 10 * 100
    if   score >= 4:  bias = "STRONG LONG ▲▲"
    elif score >= 2:  bias = "LONG BIAS   ▲"
    elif score <= -4: bias = "STRONG SHORT ▼▼"
    elif score <= -2: bias = "SHORT BIAS   ▼"
    else:             bias = "NEUTRAL / WAIT"

    # ── Suggested trade structure
    if score >= 2:
        entry  = round(price, -1)
        stop   = round(price - atr * 1.5, -1)
        tp1    = round(price + atr * 2, -1)
        tp2    = vah if vah > price else round(price + atr * 4, -1)
    elif score <= -2:
        entry  = round(price, -1)
        stop   = round(price + atr * 1.5, -1)
        tp1    = round(price - atr * 2, -1)
        tp2    = val if val < price else round(price - atr * 4, -1)
    else:
        entry = stop = tp1 = tp2 = None

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  MODULE 13 │ SIGNAL ENGINE — FINAL TRADE BIAS                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │  BIAS:        {bias:<47}│")
    print(f"  │  Score:       {score:>+3d} / 10  │  Confidence: {confidence:.0f}%              │")
    print(f"  │  Price:       ${price:>10,.1f}  │  ATR: ${atr:>8,.1f}              │")
    print(f"  │  Session:     {last['session']:<47}│")
    print(f"  └────────────────────────────────────────────────────────────┘")

    print(f"\n  Signal breakdown:")
    for r in reasons:
        print(f"    {r}")

    if entry:
        print(f"\n  ── Suggested Trade Structure ───")
        print(f"    Entry:   ${entry:>10,.0f}")
        print(f"    Stop:    ${stop:>10,.0f}  (${abs(price-stop):,.0f} risk  = {abs(price-stop)/atr:.1f}x ATR)")
        print(f"    TP1:     ${tp1:>10,.0f}  (${abs(tp1-price):,.0f} profit = {abs(tp1-price)/atr:.1f}x ATR)")
        print(f"    TP2:     ${tp2:>10,.0f}  (${abs(tp2-price):,.0f} profit = {abs(tp2-price)/atr:.1f}x ATR)")
        rr = abs(tp1-price) / abs(price-stop) if price != stop else 0
        print(f"    R:R:     {rr:.2f}  {'✓ ACCEPTABLE' if rr>=1.5 else '✗ POOR — skip'}")

    print(f"\n  ── Key Levels to Watch ───")
    print(f"    POC:    ${poc:>10,.0f}  (most traded = magnet)")
    print(f"    VAH:    ${vah:>10,.0f}  (value area high = resistance)")
    print(f"    VAL:    ${val:>10,.0f}  (value area low  = support)")
    if near_sh:
        print(f"    Stops↑: ${sorted(near_sh)[0]:>10,.0f}  (short stop cluster above)")
    if near_sl:
        print(f"    Stops↓: ${sorted(near_sl, reverse=True)[0]:>10,.0f}  (long stop cluster below)")

    return score, bias, confidence


# ══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "▓"*72)
    print("  BTC/USDT — INSTITUTIONAL ANALYSIS ENGINE")
    print("  13 Modules  |  Hedge Fund Grade  |  Binance Futures")
    print("▓"*72)

    # ── Load data ──
    dfs, funding, live = load_data()
    total = sum(len(v) for v in dfs.values())

    # ── Choose primary TF ──
    prim = "5m" if "5m" in dfs else "1m" if "1m" in dfs else list(dfs.keys())[0]
    htf  = "1h" if "1h" in dfs else prim

    print(f"\n  Primary TF: {prim}  │  HTF context: {htf}")
    df_p = build_base(dfs[prim])
    df_h = build_base(dfs[htf]) if htf != prim else df_p

    print(f"  {prim}: {len(df_p)} rows after features")

    # ══ RUN ALL MODULES ══════════════════════════════════════════

    # Module 02: CVD Pro
    df_p = cvd_pro(df_p)

    # Module 03: Big Traders
    df_p = big_traders(df_p)

    # Module 04: Order Flow
    df_p = order_flow(df_p)

    # Module 05: Unfinished Business
    unf_gaps, naked_nodes, failed_auctions = unfinished_business(df_h)

    # Module 06: VWAP/TWAP
    df_p = vwap_twap(df_p)

    # Module 07: Market Profile
    poc, vah, val, profile_df = market_profile(df_h, tick_size=25.0)

    # Module 08: Footprint
    footprint_analysis(df_p, n_candles=8, tick_size=25.0)

    # Module 09: TPO
    tpo_poc, ib_high, ib_low, single_prints = tpo_analysis(df_p, tick_size=25.0, period_minutes=30)

    # Module 10: Imbalance Chart
    buy_imbal, sell_imbal = imbalance_chart(df_p, n_candles=30, tick_size=25.0)

    # Module 11: Liquidity Map
    near_sh, near_sl, round_levels = liquidity_map(df_h)

    # Module 12: Hedge Fund Layer
    wyck_bias, fund_bias, cvd_net_bull = hedge_fund_analysis(
        df_p, poc, vah, val, funding
    )

    # Propagate some signals to df_p for signal engine
    df_p["div_bull"]       = (df_p["price_slope3"] < -0.15) & (df_p["cvd_slope3"] > 0)
    df_p["div_bear"]       = (df_p["price_slope3"] >  0.15) & (df_p["cvd_slope3"] < 0)
    df_p["absorption"]     = (df_p["vol_z"] > 1.5) & (df_p["body_pct"].abs() < 0.1)
    df_p["trapped_longs"]  = (df_p["body_pct"].shift(1) > 0.25) & (df_p["close"] < df_p["open"].shift(1))
    df_p["trapped_shorts"] = (df_p["body_pct"].shift(1) < -0.25) & (df_p["close"] > df_p["open"].shift(1))
    df_p["stacked_buy"]    = (df_p["delta_pct"] > 0.1).rolling(3).sum() == 3
    df_p["stacked_sell"]   = (df_p["delta_pct"] < -0.1).rolling(3).sum() == 3
    df_p["vwap20_dev"]     = df_p.get("vwap20_dev", pd.Series(0, index=df_p.index))

    # Module 13: Signal Engine
    score, bias, confidence = signal_engine(
        df_p, poc, vah, val, wyck_bias, fund_bias, cvd_net_bull,
        near_sh, near_sl, unf_gaps
    )

    # ══ FINAL SUMMARY ══════════════════════════════════════════════
    print("\n" + "▓"*72)
    print("  COMPLETE ANALYSIS SUMMARY")
    print("▓"*72)
    price = df_p["close"].iloc[-1]
    atr   = df_p["atr"].iloc[-1]
    print(f"""
  Price:        ${price:>12,.1f}
  ATR(14):      ${atr:>12,.1f}
  Session:      {df_p['session'].iloc[-1]}
  Live data:    {'YES ✓' if live else 'NO (synthetic — run locally for live)'}

  ── Market Structure ──────────────────────────────────────────
  POC:    ${poc:>10,.1f}   VAH:  ${vah:>10,.1f}   VAL: ${val:>10,.1f}
  IB Hi:  ${ib_high:>10,.1f}   IB Lo:${ib_low:>10,.1f}

  ── Final Signal ──────────────────────────────────────────────
  BIAS:         {bias}
  SCORE:        {score:>+d}/10   CONFIDENCE: {confidence:.0f}%

  ── Module Checklist ──────────────────────────────────────────
  ✓ CVD Pro              ✓ Big Traders          ✓ Order Flow
  ✓ Unfinished Business  ✓ VWAP/TWAP            ✓ Market Profile
  ✓ Footprint            ✓ TPO Analysis         ✓ Imbalance Chart
  ✓ Liquidity Map        ✓ Hedge Fund Layer     ✓ Signal Engine

  ── To run with LIVE Binance data ─────────────────────────────
  python btc_institutional.py
  (needs internet access to fapi.binance.com)
    """)

    print("▓"*72 + "\n")


if __name__ == "__main__":
    main()