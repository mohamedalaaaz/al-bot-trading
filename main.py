"""
BTC/USDT Binance Futures — Pattern Miner
Based on Jim Simons' "data first" approach:
  - No thesis, just find what repeats
  - Covers: time-of-day bias, funding rate behavior, session patterns, trend vs revert
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
SYMBOL     = "BTCUSDT"
INTERVAL   = "4h"          # candle size: 1m, 5m, 15m, 1h, 4h
LIMIT      = 1000        # max candles per request (Binance max = 1000)
MOVE_PCT   = 0.3           # % move threshold to count as "trend" vs "revert"

BASE_URL   = "https://fapi.binance.com"


# ─────────────────────────────────────────
#  FETCH KLINES (OHLCV)
# ─────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open","high","low","close","volume","taker_buy_vol"]:
        df[col] = df[col].astype(float)

    df["hour"]       = df["open_time"].dt.hour
    df["day_of_week"]= df["open_time"].dt.day_name()
    df["date"]       = df["open_time"].dt.date
    return df.reset_index(drop=True)


# ─────────────────────────────────────────
#  FETCH FUNDING RATE HISTORY
# ─────────────────────────────────────────
def fetch_funding_rates(symbol: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["hour"]        = df["fundingTime"].dt.hour      # always 0, 8, or 16 UTC
    return df


# ─────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Candle direction & size
    df["body_pct"]    = (df["close"] - df["open"]) / df["open"] * 100
    df["is_bull"]     = df["body_pct"] > 0
    df["range_pct"]   = (df["high"] - df["low"]) / df["open"] * 100

    # Taker delta proxy (buy pressure vs sell pressure)
    df["sell_vol"]    = df["volume"] - df["taker_buy_vol"]
    df["delta"]       = df["taker_buy_vol"] - df["sell_vol"]
    df["delta_pct"]   = df["delta"] / df["volume"]   # +1 = all buys, -1 = all sells

    # Rolling VWAP (20-period)
    typical          = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_20"]    = (typical * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["above_vwap"] = df["close"] > df["vwap_20"]

    # Next-candle outcome (what we're predicting)
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    df["next_bull"]   = df["next_return"] > 0

    # Trend vs Revert label (based on MOVE_PCT threshold)
    df["trend"]       = df["body_pct"].abs() > MOVE_PCT

    return df.dropna()


# ─────────────────────────────────────────
#  PATTERN 1: TIME-OF-DAY BIAS
# ─────────────────────────────────────────
def pattern_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("hour").agg(
        candles      =("next_bull", "count"),
        bull_rate    =("next_bull", "mean"),
        avg_move_pct =("body_pct",  "mean"),
        avg_range    =("range_pct", "mean"),
        avg_volume   =("volume",    "mean"),
    ).reset_index()

    grp["edge"] = (grp["bull_rate"] - 0.5).abs()   # distance from 50/50 = edge
    grp = grp.sort_values("edge", ascending=False)
    return grp


# ─────────────────────────────────────────
#  PATTERN 2: DAY-OF-WEEK BIAS
# ─────────────────────────────────────────
def pattern_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grp = df.groupby("day_of_week").agg(
        candles      =("next_bull", "count"),
        bull_rate    =("next_bull", "mean"),
        avg_move_pct =("body_pct",  "mean"),
        avg_range    =("range_pct", "mean"),
    ).reset_index()
    grp["day_of_week"] = pd.Categorical(grp["day_of_week"], categories=order, ordered=True)
    grp = grp.sort_values("bull_rate", ascending=False)
    return grp


# ─────────────────────────────────────────
#  PATTERN 3: DELTA DIVERGENCE
#  (price bull but delta bear, or vice versa)
# ─────────────────────────────────────────
def pattern_delta_divergence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_bull"]     = df["delta_pct"] > 0
    df["divergence"]     = df["is_bull"] != df["delta_bull"]   # price ≠ delta direction

    results = []
    for div in [True, False]:
        sub = df[df["divergence"] == div]
        results.append({
            "divergence":    "Price ≠ Delta (divergence)" if div else "Price = Delta (aligned)",
            "count":         len(sub),
            "next_bull_rate":sub["next_bull"].mean(),
            "avg_next_return":sub["next_return"].mean() * 100,
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────
#  PATTERN 4: ABOVE / BELOW VWAP
# ─────────────────────────────────────────
def pattern_vwap(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for above in [True, False]:
        sub = df[df["above_vwap"] == above]
        results.append({
            "position":       "Above VWAP" if above else "Below VWAP",
            "count":          len(sub),
            "next_bull_rate": sub["next_bull"].mean(),
            "avg_next_return":sub["next_return"].mean() * 100,
            "trend_rate":     sub["trend"].mean(),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────
#  PATTERN 5: FUNDING RATE BEHAVIOR
#  — merge funding rate into kline data
#  — check what happens after extreme funding
# ─────────────────────────────────────────
def pattern_funding(df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    # Label funding extremes
    funding = funding.copy()
    p25 = funding["fundingRate"].quantile(0.25)
    p75 = funding["fundingRate"].quantile(0.75)
    funding["regime"] = "neutral"
    funding.loc[funding["fundingRate"] >= p75, "regime"] = "high_positive"
    funding.loc[funding["fundingRate"] <= p25, "regime"] = "high_negative"

    # Merge onto klines by nearest funding time
    df = df.copy()
    df = df.sort_values("open_time")
    funding = funding.sort_values("fundingTime")

    df = pd.merge_asof(
        df, funding[["fundingTime","fundingRate","regime"]],
        left_on="open_time", right_on="fundingTime",
        direction="backward"
    )

    results = []
    for regime in ["high_positive","neutral","high_negative"]:
        sub = df[df["regime"] == regime]
        if len(sub) < 5:
            continue
        results.append({
            "funding_regime":  regime,
            "count":           len(sub),
            "avg_funding_rate":sub["fundingRate"].mean() * 100,
            "next_bull_rate":  sub["next_bull"].mean(),
            "avg_next_return": sub["next_return"].mean() * 100,
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────
#  PATTERN 6: CONSECUTIVE CANDLE STREAKS
#  — after N bull/bear candles in a row, what happens?
# ─────────────────────────────────────────
def pattern_streaks(df: pd.DataFrame, max_streak: int = 5) -> pd.DataFrame:
    df = df.copy()
    streaks = []
    current_dir = None
    count = 0

    for _, row in df.iterrows():
        d = "bull" if row["is_bull"] else "bear"
        if d == current_dir:
            count += 1
        else:
            current_dir = d
            count = 1
        streaks.append({"streak_dir": current_dir, "streak_len": min(count, max_streak)})

    df["streak_dir"] = [s["streak_dir"] for s in streaks]
    df["streak_len"] = [s["streak_len"] for s in streaks]

    results = []
    for d in ["bull","bear"]:
        for n in range(1, max_streak + 1):
            sub = df[(df["streak_dir"] == d) & (df["streak_len"] == n)]
            if len(sub) < 5:
                continue
            results.append({
                "after":           f"{n}x {d} candles",
                "count":           len(sub),
                "next_bull_rate":  sub["next_bull"].mean(),
                "avg_next_return": sub["next_return"].mean() * 100,
                "mean_revert_rate":1 - sub["next_bull"].mean() if d == "bull" else sub["next_bull"].mean(),
            })
    return pd.DataFrame(results).sort_values("mean_revert_rate", ascending=False)


# ─────────────────────────────────────────
#  PRINT HELPER
# ─────────────────────────────────────────
def print_section(title: str, df: pd.DataFrame, top_n: int = 10):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    print(df.head(top_n).to_string(index=False))


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    print(f"\n▶ Fetching {SYMBOL} {INTERVAL} klines from Binance Futures...")
    df_raw  = fetch_klines(SYMBOL, INTERVAL, LIMIT)
    print(f"  {len(df_raw)} candles loaded ({df_raw['open_time'].min().date()} → {df_raw['open_time'].max().date()})")

    print("▶ Fetching funding rate history...")
    funding = fetch_funding_rates(SYMBOL, limit=500)
    print(f"  {len(funding)} funding rate records loaded")

    print("▶ Building features...")
    df = build_features(df_raw)
    print(f"  {len(df)} usable rows after feature engineering\n")

    # ── Run all patterns ──
    tod  = pattern_time_of_day(df)
    dow  = pattern_day_of_week(df)
    delta= pattern_delta_divergence(df)
    vwap = pattern_vwap(df)
    fund = pattern_funding(df, funding)
    strk = pattern_streaks(df)

    print_section("PATTERN 1 — TIME OF DAY BIAS (sorted by edge from 50/50)", tod)
    print_section("PATTERN 2 — DAY OF WEEK BIAS", dow)
    print_section("PATTERN 3 — DELTA DIVERGENCE vs ALIGNMENT", delta)
    print_section("PATTERN 4 — ABOVE vs BELOW VWAP(20)", vwap)
    print_section("PATTERN 5 — FUNDING RATE REGIME", fund)
    print_section("PATTERN 6 — CONSECUTIVE CANDLE STREAKS (mean revert probability)", strk)

    # ── Summary: strongest edges ──
    print(f"\n{'═'*60}")
    print("  STRONGEST EDGES FOUND")
    print(f"{'═'*60}")

    # Best hour
    best_hour = tod.iloc[0]
    print(f"\n  Best Hour:     {int(best_hour['hour']):02d}:00 UTC  "
          f"→ bull_rate={best_hour['bull_rate']:.1%}  "
          f"(edge={best_hour['edge']:.3f})")

    # Best day
    best_day = dow.iloc[0]
    print(f"  Best Day:      {best_day['day_of_week']}  "
          f"→ bull_rate={best_day['bull_rate']:.1%}")

    # Delta divergence
    div_row = delta[delta["divergence"].str.contains("divergence")]
    if not div_row.empty:
        dr = div_row.iloc[0]
        print(f"  Delta Diverg:  next_bull_rate={dr['next_bull_rate']:.1%}  "
              f"avg_next_return={dr['avg_next_return']:.3f}%")

    # Best streak
    best_streak = strk.iloc[0]
    print(f"  Best Streak:   {best_streak['after']}  "
          f"→ revert_prob={best_streak['mean_revert_rate']:.1%}  "
          f"(n={int(best_streak['count'])})")

    print(f"\n{'═'*60}")
    print("  Done. No thesis. Just patterns.\n")


if __name__ == "__main__":
    main()