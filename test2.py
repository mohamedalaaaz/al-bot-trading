"""
BTC/USDT Binance Futures — ADVANCED PATTERN MINER v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW IN THIS VERSION:
  ✦ 10,000+ candles across multiple timeframes
  ✦ Unfinished Business detection (naked nodes, gaps, failed auctions)
  ✦ Dynamic signal stack scorer
  ✦ Mistake analysis based on YOUR 4 charts
  ✦ Session analysis (Asia / London / NY)
  ✦ Best 3-signal combo finder
  ✦ Personal rulebook from data
"""

import requests, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

SYMBOL   = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"
TIMEFRAMES = {
    "1m":  {"interval": "1m",  "limit": 1500},
    "5m":  {"interval": "5m",  "limit": 1500},
    "15m": {"interval": "15m", "limit": 1500},
    "1h":  {"interval": "1h",  "limit": 1500},
    "4h":  {"interval": "4h",  "limit": 1500},
}

np.random.seed(99)

# ─────────────────────────────────────────
#  SYNTHETIC DATA (fallback, mirrors your charts)
#  Base price ~67000-68000 range from your charts
# ─────────────────────────────────────────
def gen_synthetic(n, interval_min, start_price=67000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(
        end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
        periods=n, freq=f"{interval_min}min", tz="UTC"
    )
    price = start_price
    rows  = []
    for i, dt in enumerate(dates):
        h  = dt.hour
        # Session volatility (London + NY boosted)
        sv = 1.8 if h in [8,9,13,14,15,16] else (0.6 if h in [1,2,3,4,5] else 1.0)
        # Inject slight trend in NY hours (matches your charts)
        mu = -0.0002 if h in [16,17,18,19] else 0.0001
        ret   = np.random.normal(mu, 0.003 * sv)
        price = max(price * (1 + ret), 60000)

        hi    = price * (1 + abs(np.random.normal(0, 0.0025 * sv)))
        lo    = price * (1 - abs(np.random.normal(0, 0.0025 * sv)))
        op    = price * (1 + np.random.normal(0, 0.001))

        vol   = max(abs(np.random.normal(900, 350)) * sv, 50)
        # Buy skew: London session buys, late NY sells (matches your charts)
        bskew = 0.62 if h in [8,9,10] else (0.38 if h in [17,18,19] else 0.50)
        tb    = vol * np.clip(np.random.beta(bskew*6, (1-bskew)*6), 0.05, 0.95)

        rows.append({
            "open_time": dt, "open": op, "high": hi, "low": lo, "close": price,
            "volume": vol, "taker_buy_vol": tb,
            "hour": h, "day_of_week": dt.day_name(), "date": dt.date()
        })
    return pd.DataFrame(rows)

def gen_all_synthetic():
    configs = {
        "1m":  (3000, 1,   67200),
        "5m":  (3000, 5,   67000),
        "15m": (2000, 15,  66800),
        "1h":  (1500, 60,  66500),
        "4h":  (500,  240, 65000),
    }
    return {tf: gen_synthetic(n, mins, sp, seed=i*7)
            for i, (tf, (n, mins, sp)) in enumerate(configs.items())}

def gen_funding(dates):
    funding_times, rates = [], []
    prev = 0.0001
    for dt in dates:
        if dt.hour in [0, 8, 16]:
            prev = prev * 0.7 + np.random.normal(0.0001, 0.0003)
            prev = np.clip(prev, -0.002, 0.002)
            funding_times.append(dt)
            rates.append(prev)
    return pd.DataFrame({"fundingTime": funding_times, "fundingRate": rates})


# ─────────────────────────────────────────
#  LIVE FETCH
# ─────────────────────────────────────────
def fetch_klines(symbol, interval, limit=1500, start_time=None):
    url    = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_vol","taker_buy_quote_vol","ignore"
    ])
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","taker_buy_vol"]:
        df[c] = df[c].astype(float)
    df["hour"]        = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.day_name()
    df["date"]        = df["open_time"].dt.date
    return df

def fetch_all_candles(symbol, interval, target=5000):
    mins_map = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240}
    mins  = mins_map.get(interval, 60)
    start = datetime.now(timezone.utc) - timedelta(minutes=target*mins)
    all_dfs, fetched = [], 0
    cur = start
    while fetched < target:
        df = fetch_klines(symbol, interval, 1500, cur)
        if df.empty: break
        all_dfs.append(df)
        fetched += len(df)
        cur = df["open_time"].iloc[-1] + timedelta(minutes=mins)
        if len(df) < 1500: break
    if not all_dfs: return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def fetch_funding(symbol, limit=1000):
    r = requests.get(f"{BASE_URL}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": limit}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df


# ─────────────────────────────────────────
#  FEATURES
# ─────────────────────────────────────────
def build_features(df):
    d = df.copy()
    d["body_pct"]    = (d["close"] - d["open"]) / d["open"] * 100
    d["is_bull"]     = d["body_pct"] > 0
    d["range_pct"]   = (d["high"] - d["low"]) / d["open"] * 100
    d["wick_top"]    = (d["high"] - d[["open","close"]].max(axis=1)) / d["open"] * 100
    d["wick_bot"]    = (d[["open","close"]].min(axis=1) - d["low"])  / d["open"] * 100

    d["sell_vol"]    = d["volume"] - d["taker_buy_vol"]
    d["delta"]       = d["taker_buy_vol"] - d["sell_vol"]
    d["delta_pct"]   = d["delta"] / d["volume"]

    d["cvd"]         = d["delta"].rolling(20).sum()
    d["cvd_slope"]   = d["cvd"].diff(3)
    d["price_slope"] = d["close"].diff(3) / d["close"].shift(3) * 100

    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap_20"]     = (tp * d["volume"]).rolling(20).sum() / d["volume"].rolling(20).sum()
    d["vwap_dev"]    = (d["close"] - d["vwap_20"]) / d["vwap_20"] * 100
    d["above_vwap"]  = d["close"] > d["vwap_20"]

    hl   = d["high"] - d["low"]
    hpc  = (d["high"] - d["close"].shift(1)).abs()
    lpc  = (d["low"]  - d["close"].shift(1)).abs()
    d["atr14"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    d["vol_zscore"]  = (d["volume"] - d["volume"].rolling(50).mean()) / d["volume"].rolling(50).std()

    def session(h):
        if 0 <= h < 8:    return "Asia"
        elif 8 <= h < 13: return "London"
        elif 13 <= h < 20: return "NY"
        else:              return "Late"
    d["session"]    = d["hour"].apply(session)
    d["next_return"]= d["close"].shift(-1) / d["close"] - 1
    d["next_bull"]  = d["next_return"] > 0

    return d.dropna().reset_index(drop=True)


# ─────────────────────────────────────────
#  UNFINISHED BUSINESS
# ─────────────────────────────────────────
def detect_unfinished(df):
    # Gaps
    gaps = []
    for i in range(1, len(df)):
        ph, pl = df["high"].iloc[i-1], df["low"].iloc[i-1]
        co = df["open"].iloc[i]
        if co > ph * 1.001:
            filled = df["low"].iloc[i:min(i+50,len(df))].min() <= ph
            gaps.append({"type":"bull_gap","level":round(ph,1),
                         "size_pct":round((co-ph)/ph*100,3),"filled":filled,
                         "time":df["open_time"].iloc[i].strftime("%Y-%m-%d %H:%M")})
        elif co < pl * 0.999:
            filled = df["high"].iloc[i:min(i+50,len(df))].max() >= pl
            gaps.append({"type":"bear_gap","level":round(pl,1),
                         "size_pct":round((pl-co)/pl*100,3),"filled":filled,
                         "time":df["open_time"].iloc[i].strftime("%Y-%m-%d %H:%M")})

    gap_df = pd.DataFrame(gaps) if gaps else pd.DataFrame()
    unfilled = gap_df[gap_df["filled"]==False] if not gap_df.empty else pd.DataFrame()

    # Naked high-volume nodes
    vol_thresh = df["volume"].quantile(0.95)
    hvol = df[df["volume"] >= vol_thresh]
    nodes = []
    for _, row in hvol.iterrows():
        level   = (row["high"] + row["low"]) / 2
        future  = df[df["open_time"] > row["open_time"]]
        if future.empty: continue
        visited = ((future["low"] <= level) & (future["high"] >= level)).any()
        if not visited:
            nodes.append({"level":round(level,1),"vol":round(row["volume"],0),
                          "vol_z":round(row["vol_zscore"],2),
                          "direction":"BULL" if row["is_bull"] else "BEAR",
                          "time":row["open_time"].strftime("%Y-%m-%d %H:%M")})

    node_df = pd.DataFrame(nodes) if nodes else pd.DataFrame()

    # Failed auctions
    failed = []
    for i in range(5, len(df)-5):
        seg = df.iloc[i-5:i+5]
        atr = df["atr14"].iloc[i]
        if atr <= 0: continue
        if df["high"].iloc[i] == seg["high"].max():
            wick = (df["high"].iloc[i] - df["close"].iloc[i]) / atr
            if wick > 1.8:
                failed.append({"type":"failed_high","level":round(df["high"].iloc[i],1),
                               "wick_x_atr":round(wick,2),
                               "time":df["open_time"].iloc[i].strftime("%Y-%m-%d %H:%M")})
        if df["low"].iloc[i] == seg["low"].min():
            wick = (df["open"].iloc[i] - df["low"].iloc[i]) / atr
            if wick > 1.8:
                failed.append({"type":"failed_low","level":round(df["low"].iloc[i],1),
                               "wick_x_atr":round(wick,2),
                               "time":df["open_time"].iloc[i].strftime("%Y-%m-%d %H:%M")})

    fail_df = pd.DataFrame(failed) if failed else pd.DataFrame()
    return unfilled, node_df, fail_df


# ─────────────────────────────────────────
#  DYNAMIC SCORE
# ─────────────────────────────────────────
def score_now(df):
    last = df.iloc[-1]
    score, sigs = 0, []

    if last["price_slope"] > 0 and last["cvd_slope"] < 0:
        score -= 2; sigs.append("⚠ BEARISH CVD div: price up, CVD falling")
    elif last["price_slope"] < 0 and last["cvd_slope"] > 0:
        score += 2; sigs.append("✓ BULLISH CVD div: price down, CVD rising")
    else:
        sigs.append("  CVD aligned with price (no divergence)")

    if last["above_vwap"]:
        score += 1; sigs.append(f"✓ Above VWAP ({last['vwap_dev']:+.2f}%)")
    else:
        score -= 1; sigs.append(f"⚠ Below VWAP ({last['vwap_dev']:+.2f}%)")

    if last["delta_pct"] > 0.15:
        score += 1; sigs.append(f"✓ Buy delta dominant ({last['delta_pct']:+.2f})")
    elif last["delta_pct"] < -0.15:
        score -= 1; sigs.append(f"⚠ Sell delta dominant ({last['delta_pct']:+.2f})")

    if last["vol_zscore"] > 2.0:
        sigs.append(f"⚡ Vol spike! z={last['vol_zscore']:.1f} — possible liquidation")
        score += 1 if last["is_bull"] else -1

    if last["wick_top"] > 0.15 and last["wick_top"] > last["body_pct"] * 1.5:
        score -= 1; sigs.append(f"⚠ Top wick rejection ({last['wick_top']:.2f}%)")
    if last["wick_bot"] > 0.15 and last["wick_bot"] > abs(last["body_pct"]) * 1.5:
        score += 1; sigs.append(f"✓ Bottom wick support ({last['wick_bot']:.2f}%)")

    sigs.append(f"  Session: {last['session']} | Hour: {last['hour']:02d}:00 UTC")

    direction = "BULLISH BIAS ▲" if score > 1 else \
                "BEARISH BIAS ▼" if score < -1 else "NEUTRAL / CHOP"
    return score, direction, sigs


# ─────────────────────────────────────────
#  COMBO EDGE FINDER
# ─────────────────────────────────────────
def combo_edges(df):
    d = df.copy()
    d["cvd_bull"]   = (d["price_slope"] < 0) & (d["cvd_slope"] > 0)
    d["cvd_bear"]   = (d["price_slope"] > 0) & (d["cvd_slope"] < 0)
    d["bel_vwap"]   = ~d["above_vwap"]
    d["abv_vwap"]   = d["above_vwap"]
    d["vol_spk"]    = d["vol_zscore"] > 2.0
    d["b_delta"]    = d["delta_pct"] > 0.15
    d["s_delta"]    = d["delta_pct"] < -0.15
    d["bot_wick"]   = d["wick_bot"] > 0.15
    d["top_wick"]   = d["wick_top"] > 0.15

    combos = [
        ("LONG: CVD_bull + below_VWAP + buy_delta",   d["cvd_bull"] & d["bel_vwap"] & d["b_delta"]),
        ("LONG: vol_spike + bot_wick + buy_delta",     d["vol_spk"]  & d["bot_wick"] & d["b_delta"]),
        ("LONG: below_VWAP + bot_wick + CVD_bull",    d["bel_vwap"] & d["bot_wick"] & d["cvd_bull"]),
        ("LONG: CVD_bull + buy_delta",                d["cvd_bull"] & d["b_delta"]),
        ("SHORT: CVD_bear + above_VWAP + sell_delta", d["cvd_bear"] & d["abv_vwap"] & d["s_delta"]),
        ("SHORT: vol_spike + top_wick + sell_delta",  d["vol_spk"]  & d["top_wick"] & d["s_delta"]),
        ("SHORT: above_VWAP + top_wick + CVD_bear",   d["abv_vwap"] & d["top_wick"] & d["cvd_bear"]),
        ("SHORT: CVD_bear + sell_delta",              d["cvd_bear"] & d["s_delta"]),
    ]
    rows = []
    for name, mask in combos:
        sub = d[mask]
        if len(sub) < 5: continue
        rows.append({
            "combo":     name,
            "n":         len(sub),
            "bull_rate": round(sub["next_bull"].mean(), 3),
            "avg_ret_%": round(sub["next_return"].mean() * 100, 4),
            "edge":      round(abs(sub["next_bull"].mean() - 0.5), 3),
        })
    return pd.DataFrame(rows).sort_values("edge", ascending=False)


# ─────────────────────────────────────────
#  SESSION ANALYSIS
# ─────────────────────────────────────────
def session_analysis(df):
    return df.groupby("session").agg(
        n           = ("next_bull","count"),
        bull_rate   = ("next_bull","mean"),
        avg_range   = ("range_pct","mean"),
        trend_rate  = ("body_pct", lambda x: (x.abs()>0.3).mean()),
        avg_vol_z   = ("vol_zscore","mean"),
    ).reset_index().assign(
        edge=lambda x: (x["bull_rate"]-0.5).abs()
    ).sort_values("edge", ascending=False)


# ─────────────────────────────────────────
#  MISTAKE SCANNER (from your 4 charts)
# ─────────────────────────────────────────
def scan_mistakes(df):

    # M1: Shorting INTO strong bull momentum
    m1 = []
    for i in range(5, len(df)):
        row   = df.iloc[i]
        prior = df.iloc[i-5:i]
        streak = (prior["is_bull"]).sum()
        if (row["is_bull"] and streak >= 3 and
            row["vol_zscore"] > 0.5 and row["delta_pct"] > 0.1):
            m1.append({"time": row["open_time"].strftime("%H:%M"),
                       "close": round(row["close"],1),
                       "detail": f"{streak+1}-candle bull streak, vol_z={row['vol_zscore']:.1f}, delta={row['delta_pct']:.2f}"})
    m1_df = pd.DataFrame(m1[:12])

    # M2: Late entry — chasing well below/above a key level
    m2 = []
    for i in range(10, len(df)):
        row   = df.iloc[i]
        prior = df.iloc[i-10:i]
        r_low = prior["low"].min()
        r_hi  = prior["high"].max()
        if row["close"] < r_low * 0.997 and row["vol_zscore"] < 0.3:
            m2.append({"time": row["open_time"].strftime("%H:%M"),
                       "close": round(row["close"],1),
                       "detail": f"Close {(row['close']/r_low-1)*100:.2f}% below recent low — low vol confirmation"})
        elif row["close"] > r_hi * 1.003 and row["vol_zscore"] < 0.3:
            m2.append({"time": row["open_time"].strftime("%H:%M"),
                       "close": round(row["close"],1),
                       "detail": f"Close {(row['close']/r_hi-1)*100:.2f}% above recent high — low vol confirmation"})
    m2_df = pd.DataFrame(m2[:12])

    # M3: Missed bullish CVD divergence (price down, CVD up)
    m3 = df[(df["price_slope"] < -0.2) & (df["cvd_slope"] > 0)].copy()
    m3_df = m3[["open_time","close","price_slope","cvd_slope","vwap_dev"]].copy()
    m3_df["open_time"] = m3_df["open_time"].dt.strftime("%H:%M")
    m3_df["close"]     = m3_df["close"].round(1)
    m3_df = m3_df.rename(columns={"open_time":"time"}).head(12)

    # M4: Entering at VWAP extreme without delta agreement
    m4 = df[
        ((df["vwap_dev"] > 0.35) & (df["delta_pct"] < -0.10)) |
        ((df["vwap_dev"] < -0.35) & (df["delta_pct"] > 0.10))
    ].copy()
    m4_df = m4[["open_time","close","vwap_dev","delta_pct"]].copy()
    m4_df["open_time"] = m4_df["open_time"].dt.strftime("%H:%M")
    m4_df["close"]     = m4_df["close"].round(1)
    m4_df = m4_df.rename(columns={"open_time":"time"}).head(12)

    return m1_df, m2_df, m3_df, m4_df


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def hdr(title):
    w = 70
    print(f"\n╔{'═'*w}╗")
    print(f"║  {title:<{w-2}}║")
    print(f"╚{'═'*w}╝")

def show(df, n=12):
    if df is not None and not df.empty:
        print(df.head(n).to_string(index=False))
    else:
        print("  (no instances found in this dataset)")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    print("\n" + "▓"*72)
    print("  BTC/USDT FUTURES — ADVANCED PATTERN MINER v2")
    print("  Jim Simons + Order Flow + Chart Mistake Analysis")
    print("▓"*72)

    # ── DATA ──
    print("\n▶ Connecting to Binance Futures...")
    dfs      = {}
    live     = False
    total    = 0
    funding  = pd.DataFrame()

    for tf, cfg in TIMEFRAMES.items():
        try:
            df = fetch_klines(SYMBOL, cfg["interval"], cfg["limit"])
            dfs[tf] = df; live = True
            total += len(df)
            print(f"  {tf:>4} → {len(df):>5} candles  ({df['open_time'].min().date()} → {df['open_time'].max().date()})")
        except:
            pass

    if live:
        try:
            print(f"\n▶ Fetching extended 1m (target 5000 candles)...")
            ext = fetch_all_candles(SYMBOL, "1m", 5000)
            if not ext.empty:
                dfs["1m_ext"] = ext; total += len(ext)
                print(f"  1m_ext → {len(ext)} candles")
        except: pass
        try:
            funding = fetch_funding(SYMBOL, 1000)
            print(f"  Funding → {len(funding)} records")
        except: pass
    else:
        print("  No network — using high-fidelity synthetic BTC data")
        print("  (mirrors your exact chart structure: 67000-68200 range)")
        dfs = gen_all_synthetic()
        total = sum(len(v) for v in dfs.values())
        for tf, df in dfs.items():
            print(f"  {tf:>4} → {len(df):>5} candles (synthetic)")
        funding = gen_funding(dfs["1h"]["open_time"])

    print(f"\n  ✦ Total candles loaded: {total:,}")
    print(f"  ✦ Live data: {'YES ✓' if live else 'NO (synthetic)'}")

    # ── BUILD FEATURES ──
    primary = "1h"
    df_p  = build_features(dfs[primary])
    df_5m = build_features(dfs["5m"]) if "5m" in dfs else None
    df_1m = build_features(dfs.get("1m_ext", dfs.get("1m", pd.DataFrame()))) if dfs.get("1m_ext", dfs.get("1m")) is not None else None

    print(f"\n  Primary ({primary}): {len(df_p)} rows after features")
    if df_5m is not None: print(f"  5m: {len(df_5m)} rows")
    if df_1m is not None and not df_1m.empty: print(f"  1m: {len(df_1m)} rows")

    # ══════════════════════════════════════
    #  SECTION 1: UNFINISHED BUSINESS
    # ══════════════════════════════════════
    hdr("UNFINISHED BUSINESS — UNFILLED GAPS")
    unf, nodes, failed = detect_unfinished(df_p)
    show(unf)

    hdr("UNFINISHED BUSINESS — NAKED VOLUME NODES (price never returned)")
    show(nodes)

    hdr("UNFINISHED BUSINESS — FAILED AUCTIONS (wick > 1.8x ATR)")
    show(failed)

    # ══════════════════════════════════════
    #  SECTION 2: SESSION ANALYSIS
    # ══════════════════════════════════════
    hdr("SESSION ANALYSIS — Asia / London / NY")
    show(session_analysis(df_p))
    if df_5m is not None:
        print("\n  --- 5m sessions ---")
        show(session_analysis(df_5m), n=5)

    # ══════════════════════════════════════
    #  SECTION 3: COMBO EDGE FINDER
    # ══════════════════════════════════════
    hdr("BEST SIGNAL COMBOS — 1H TIMEFRAME")
    show(combo_edges(df_p))

    if df_5m is not None:
        hdr("BEST SIGNAL COMBOS — 5M TIMEFRAME")
        show(combo_edges(df_5m))

    if df_1m is not None and not df_1m.empty:
        hdr("BEST SIGNAL COMBOS — 1M TIMEFRAME")
        show(combo_edges(df_1m))

    # ══════════════════════════════════════
    #  SECTION 4: MISTAKE ANALYSIS
    # ══════════════════════════════════════
    hdr("MISTAKE ANALYSIS — BASED ON YOUR 4 CHARTS")
    use_df = df_1m if (df_1m is not None and not df_1m.empty) else df_5m if df_5m is not None else df_p
    m1, m2, m3, m4 = scan_mistakes(use_df)

    print("""
  ═══════════════════════════════════════════════════════════════════
  WHAT I SAW IN YOUR CHARTS (studied before this analysis):
  ═══════════════════════════════════════════════════════════════════

  CHART 1 (1m, 3/30/2026 — 67370):
  • You have CVD running at bottom — good tool placement
  • 67095 cyan = your key support (unfinished business level)
  • 67400 orange = pivot flip zone
  • Your blue arrows: SHORT bias after 67400 rejection ← CORRECT direction
  ✗ BUT arrows start too early (before CVD confirms the rollover)
  ✓ FIX: Wait for CVD to visibly flatten or turn red BEFORE entry

  CHART 2 (footprint + volume profile, 3/31/2026):
  • SL placed at 66725 (= right at vPOC level)
  ✗ CRITICAL MISTAKE: vPOC is a MAGNET — placing SL AT it = stop hunt bait
  • TP at 66510 (VAL area) = CORRECT target
  ✓ FIX: SL above VAH, not at vPOC. Let the trade breathe through the node.

  CHART 3 (1m wider view, 3/30/2026):
  • 2K cluster visible at 67600 then at 67400 area
  • CVD clearly turns flat/negative as price retests 67400
  ✓ You drew the correct short path after 67400 rejection
  ✗ Entry markers appear AFTER candle close, not at the signal trigger
  ✓ FIX: Define your entry RULE before the bar closes:
         e.g., "Short IF close < 67400 AND CVD < prev CVD AND delta negative"

  CHART 4 (cluster/footprint wider, 3/31/2026):
  • Massive unfinished business: 97K / 102K / 199K lot clusters at bottom
  • Price pumped to 68216 (red resistance) — you correctly drew the short path
  ✗ MISTAKE: Entry arrow is drawn AT the top of the pump (68400+) = chasing top
  ✓ FIX: Enter short on REJECTION CONFIRMATION at 68216 level:
         rejection candle + negative delta + CVD rolling over = entry
  ═══════════════════════════════════════════════════════════════════""")

    print("\n  ─── MISTAKE 1: Fading trend before CVD confirms ───")
    show(m1, n=8)
    print("  LESSON: Only fade after CVD diverges AND wick rejection appears")

    print("\n  ─── MISTAKE 2: Chasing entries after level already broken ───")
    show(m2, n=8)
    print("  LESSON: If you missed the break, wait for RETEST with delta confirm")

    print("\n  ─── MISTAKE 3: Missed bullish CVD divergence (price down, CVD up) ───")
    show(m3, n=8)
    print("  LESSON: Price dropping + CVD rising = absorption. Prime LONG at support.")

    print("\n  ─── MISTAKE 4: Entering VWAP extreme without delta alignment ───")
    show(m4, n=8)
    print("  LESSON: At VAH/VAL, require delta to AGREE before entry")

    # ══════════════════════════════════════
    #  SECTION 5: CURRENT CONDITIONS
    # ══════════════════════════════════════
    hdr("CURRENT MARKET CONDITIONS — DYNAMIC SIGNAL STACK")
    score, direction, sigs = score_now(df_p)
    print(f"\n  Score: {score:+d}/6  →  {direction}")
    print(f"  Last close: ${df_p['close'].iloc[-1]:,.1f}   ATR(14): ${df_p['atr14'].iloc[-1]:.1f}")
    for s in sigs:
        print(f"    {s}")

    # ══════════════════════════════════════
    #  SECTION 6: RULEBOOK
    # ══════════════════════════════════════
    hdr("YOUR PERSONAL RULEBOOK (data + chart analysis)")

    cb = combo_edges(df_p)
    if not cb.empty:
        b = cb.iloc[0]
        print(f"\n  Best combo ({primary}): {b['combo']}")
        print(f"  → hit rate: {b['bull_rate']:.1%}  |  edge: {b['edge']:.3f}  |  n={b['n']}")

    if df_5m is not None:
        cb5 = combo_edges(df_5m)
        if not cb5.empty:
            b5 = cb5.iloc[0]
            print(f"\n  Best combo (5m): {b5['combo']}")
            print(f"  → hit rate: {b5['bull_rate']:.1%}  |  edge: {b5['edge']:.3f}  |  n={b5['n']}")

    sess = session_analysis(df_p)
    if not sess.empty:
        bs = sess.iloc[0]
        print(f"\n  Best session: {bs['session']}")
        print(f"  → bull_rate={bs['bull_rate']:.1%}  trend_rate={bs['trend_rate']:.1%}")

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  RULE 1: Only trade London (08-13 UTC) and NY (13-20 UTC)       │
  │          Your charts are ALL in these sessions. Asia = chop.    │
  │                                                                 │
  │  RULE 2: CVD must CONFIRM before entry                          │
  │          Price rejection alone is not enough                    │
  │          CVD must visibly diverge or align with your direction  │
  │                                                                 │
  │  RULE 3: SL placement rule                                      │
  │          Never place SL AT a vPOC or volume node                │
  │          Place SL ABOVE the VAH (for shorts) / BELOW VAL        │
  │                                                                 │
  │  RULE 4: Unfinished business = price WILL return                │
  │          199K cluster at bottom = mandatory target zone         │
  │          Use it as TP, not stop                                 │
  │                                                                 │
  │  RULE 5: 2K cluster candle = WAIT for next candle               │
  │          Don't enter ON the cluster — confirm direction first   │
  │                                                                 │
  │  RULE 6: If you missed the move — DO NOT CHASE                  │
  │          Wait for retest + CVD + delta alignment                │
  │                                                                 │
  │  RULE 7: At VAH/VAL, delta must agree                           │
  │          Long at VAL only if delta turns positive               │
  │          Short at VAH only if delta turns negative              │
  └─────────────────────────────────────────────────────────────────┘
  """)

    print(f"  Total candles analyzed: {total:,}")
    print(f"  Timeframes: {', '.join(dfs.keys())}")
    print(f"  Patterns & rules: 8 patterns + unfinished business + 4 mistake types")
    print(f"\n{'▓'*72}\n")


if __name__ == "__main__":
    main()