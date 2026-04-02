"""
BTC/USDT Binance Futures — Full Pattern Miner
Includes:
  1. Time-of-day bias
  2. Day-of-week bias
  3. Delta / CVD divergence
  4. VWAP position
  5. Funding rate regime
  6. Candle streaks
  7. Liquidation cluster detection  ← NEW
  8. CVD divergence scoring         ← NEW
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────
#  GENERATE REALISTIC SYNTHETIC BTC DATA
#  (mimics Binance Futures kline + funding structure)
# ─────────────────────────────────────────
def generate_btc_data(n=1000):
    dates = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")
    price = 42000.0
    prices, volumes, taker_buys = [], [], []

    for i in range(n):
        hour = dates[i].hour
        # Inject time-of-day volatility bias (London + NY sessions)
        session_vol = 1.5 if hour in [8,9,13,14,15,16,17] else 0.8
        ret = np.random.normal(0.0001, 0.0035 * session_vol)
        price *= (1 + ret)
        vol = abs(np.random.normal(800, 250)) * (1 + 0.5 * session_vol)
        # Inject delta bias by hour
        buy_skew = 0.60 if hour in [13,14,15] else (0.40 if hour in [2,3,4] else 0.50)
        tb = vol * np.random.beta(buy_skew * 5, (1 - buy_skew) * 5)
        prices.append(price)
        volumes.append(max(vol, 10))
        taker_buys.append(max(tb, 1))

    prices = np.array(prices)
    noise  = np.random.normal(0, 0.001, n)

    df = pd.DataFrame({
        "open_time":      dates,
        "open":           prices * (1 + np.random.normal(0, 0.0005, n)),
        "high":           prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
        "low":            prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
        "close":          prices,
        "volume":         volumes,
        "taker_buy_vol":  taker_buys,
    })
    df["hour"]        = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.day_name()
    df["date"]        = df["open_time"].dt.date
    return df

def generate_funding_rates(dates):
    # Funding every 8h at 0, 8, 16 UTC
    funding_times = []
    rates = []
    base = dates[0].normalize()
    t = base
    while t <= dates.iloc[-1]:
        if t.hour in [0, 8, 16]:
            funding_times.append(t)
            # Slightly autocorrelated funding
            prev = rates[-1] if rates else 0.0001
            r = prev * 0.7 + np.random.normal(0.0001, 0.0003)
            rates.append(np.clip(r, -0.002, 0.002))
        t += pd.Timedelta(hours=1)
    return pd.DataFrame({"fundingTime": funding_times, "fundingRate": rates})


# ─────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df["body_pct"]    = (df["close"] - df["open"]) / df["open"] * 100
    df["is_bull"]     = df["body_pct"] > 0
    df["range_pct"]   = (df["high"] - df["low"]) / df["open"] * 100
    df["sell_vol"]    = df["volume"] - df["taker_buy_vol"]
    df["delta"]       = df["taker_buy_vol"] - df["sell_vol"]
    df["delta_pct"]   = df["delta"] / df["volume"]

    # Rolling CVD (cumulative volume delta, 20-period)
    df["cvd"]         = df["delta"].rolling(20).sum()
    df["cvd_slope"]   = df["cvd"].diff(3)   # CVD momentum last 3 bars
    df["price_slope"] = df["close"].diff(3) / df["close"].shift(3) * 100

    # VWAP(20)
    typical           = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_20"]     = (typical * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["vwap_dev"]    = (df["close"] - df["vwap_20"]) / df["vwap_20"] * 100
    df["above_vwap"]  = df["close"] > df["vwap_20"]

    # Volume spike (potential liquidation zone)
    df["vol_zscore"]  = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()

    # Next candle outcome
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    df["next_bull"]   = df["next_return"] > 0
    df["trend"]       = df["body_pct"].abs() > 0.3

    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────
#  PATTERN 7: LIQUIDATION CLUSTER DETECTION
#  Volume spikes = likely liquidation events
#  What happens in the 1–3 candles AFTER?
# ─────────────────────────────────────────
def pattern_liquidation_clusters(df):
    df = df.copy()

    # Classify spike strength
    df["liq_event"] = "normal"
    df.loc[df["vol_zscore"] > 1.5,  "liq_event"] = "moderate_spike"
    df.loc[df["vol_zscore"] > 2.5,  "liq_event"] = "strong_spike"
    df.loc[df["vol_zscore"] > 3.5,  "liq_event"] = "extreme_spike"

    # Also split by direction (bull liquidation vs bear liquidation)
    df["liq_type"] = "none"
    mask_spike = df["vol_zscore"] > 2.0
    df.loc[mask_spike &  df["is_bull"], "liq_type"] = "bull_liq_candle"   # shorts liquidated
    df.loc[mask_spike & ~df["is_bull"], "liq_type"] = "bear_liq_candle"   # longs liquidated

    results = []
    for evt in ["moderate_spike","strong_spike","extreme_spike"]:
        sub = df[df["liq_event"] == evt]
        if len(sub) < 3:
            continue
        results.append({
            "event":            evt,
            "count":            len(sub),
            "next_bull_rate":   sub["next_bull"].mean(),
            "avg_next_ret_%":   round(sub["next_return"].mean() * 100, 4),
            "avg_vol_zscore":   round(sub["vol_zscore"].mean(), 2),
            "bull_candle_%":    round(sub["is_bull"].mean() * 100, 1),
        })

    for ltype in ["bull_liq_candle","bear_liq_candle"]:
        sub = df[df["liq_type"] == ltype]
        if len(sub) < 3:
            continue
        results.append({
            "event":            ltype,
            "count":            len(sub),
            "next_bull_rate":   sub["next_bull"].mean(),
            "avg_next_ret_%":   round(sub["next_return"].mean() * 100, 4),
            "avg_vol_zscore":   round(sub["vol_zscore"].mean(), 2),
            "bull_candle_%":    round(sub["is_bull"].mean() * 100, 1),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
#  PATTERN 8: CVD DIVERGENCE SCORING
#  Score each candle 0–4 based on how many
#  divergence signals are present
# ─────────────────────────────────────────
def pattern_cvd_divergence(df):
    df = df.copy()

    # Individual divergence signals
    df["sig_price_bull_cvd_bear"] = (df["price_slope"] > 0) & (df["cvd_slope"] < 0)
    df["sig_price_bear_cvd_bull"] = (df["price_slope"] < 0) & (df["cvd_slope"] > 0)
    df["sig_above_vwap_neg_delta"]= df["above_vwap"] & (df["delta_pct"] < -0.1)
    df["sig_below_vwap_pos_delta"]= ~df["above_vwap"] & (df["delta_pct"] > 0.1)

    # Divergence score (0–4)
    df["div_score"] = (
        df["sig_price_bull_cvd_bear"].astype(int) +
        df["sig_price_bear_cvd_bull"].astype(int) +
        df["sig_above_vwap_neg_delta"].astype(int) +
        df["sig_below_vwap_pos_delta"].astype(int)
    )

    # Bearish divergence: price up, internals weak
    df["bear_div_score"] = (
        df["sig_price_bull_cvd_bear"].astype(int) +
        df["sig_above_vwap_neg_delta"].astype(int)
    )
    # Bullish divergence: price down, internals strong
    df["bull_div_score"] = (
        df["sig_price_bear_cvd_bull"].astype(int) +
        df["sig_below_vwap_pos_delta"].astype(int)
    )

    results = []
    for score in sorted(df["div_score"].unique()):
        sub = df[df["div_score"] == score]
        results.append({
            "div_score":      int(score),
            "count":          len(sub),
            "next_bull_rate": round(sub["next_bull"].mean(), 3),
            "avg_next_ret_%": round(sub["next_return"].mean() * 100, 4),
            "label":          "no divergence" if score == 0 else
                              "weak div" if score == 1 else
                              "moderate div" if score == 2 else "STRONG div",
        })

    # High-confidence bearish divergence
    bear_high = df[df["bear_div_score"] == 2]
    bull_high = df[df["bull_div_score"] == 2]

    print(f"\n  ► High-confidence BEARISH divergence (score=2): {len(bear_high)} setups")
    if len(bear_high) > 0:
        print(f"    → next_bull_rate : {bear_high['next_bull'].mean():.1%}")
        print(f"    → avg_next_return: {bear_high['next_return'].mean()*100:.4f}%")
        print(f"    → interpretation : {'bearish edge confirmed ✓' if bear_high['next_bull'].mean() < 0.45 else 'weak edge'}")

    print(f"\n  ► High-confidence BULLISH divergence (score=2): {len(bull_high)} setups")
    if len(bull_high) > 0:
        print(f"    → next_bull_rate : {bull_high['next_bull'].mean():.1%}")
        print(f"    → avg_next_return: {bull_high['next_return'].mean()*100:.4f}%")
        print(f"    → interpretation : {'bullish edge confirmed ✓' if bull_high['next_bull'].mean() > 0.55 else 'weak edge'}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────
#  ORIGINAL PATTERNS (1–6)
# ─────────────────────────────────────────
def pattern_time_of_day(df):
    grp = df.groupby("hour").agg(
        candles      =("next_bull","count"),
        bull_rate    =("next_bull","mean"),
        avg_move_pct =("body_pct","mean"),
        avg_range    =("range_pct","mean"),
    ).reset_index()
    grp["edge"] = (grp["bull_rate"] - 0.5).abs()
    return grp.sort_values("edge", ascending=False)

def pattern_day_of_week(df):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grp = df.groupby("day_of_week").agg(
        candles   =("next_bull","count"),
        bull_rate =("next_bull","mean"),
        avg_range =("range_pct","mean"),
    ).reset_index()
    return grp.sort_values("bull_rate", ascending=False)

def pattern_vwap(df):
    results = []
    for above in [True, False]:
        sub = df[df["above_vwap"] == above]
        results.append({
            "position":       "Above VWAP" if above else "Below VWAP",
            "count":          len(sub),
            "next_bull_rate": round(sub["next_bull"].mean(), 3),
            "avg_next_ret_%": round(sub["next_return"].mean()*100, 4),
        })
    return pd.DataFrame(results)

def pattern_funding(df, funding):
    p25 = funding["fundingRate"].quantile(0.25)
    p75 = funding["fundingRate"].quantile(0.75)
    funding = funding.copy()
    funding["regime"] = "neutral"
    funding.loc[funding["fundingRate"] >= p75, "regime"] = "high_positive"
    funding.loc[funding["fundingRate"] <= p25, "regime"] = "high_negative"
    df = df.copy().sort_values("open_time")
    funding = funding.sort_values("fundingTime")
    df = pd.merge_asof(df, funding[["fundingTime","fundingRate","regime"]],
                       left_on="open_time", right_on="fundingTime", direction="backward")
    results = []
    for regime in ["high_positive","neutral","high_negative"]:
        sub = df[df["regime"] == regime]
        if len(sub) < 5: continue
        results.append({
            "funding_regime":  regime,
            "count":           len(sub),
            "avg_funding_%":   round(sub["fundingRate"].mean()*100, 4),
            "next_bull_rate":  round(sub["next_bull"].mean(), 3),
            "avg_next_ret_%":  round(sub["next_return"].mean()*100, 4),
        })
    return pd.DataFrame(results)

def pattern_streaks(df, max_streak=5):
    df = df.copy()
    streaks, current_dir, count = [], None, 0
    for _, row in df.iterrows():
        d = "bull" if row["is_bull"] else "bear"
        count = count + 1 if d == current_dir else 1
        current_dir = d
        streaks.append({"streak_dir": current_dir, "streak_len": min(count, max_streak)})
    df["streak_dir"] = [s["streak_dir"] for s in streaks]
    df["streak_len"] = [s["streak_len"] for s in streaks]
    results = []
    for d in ["bull","bear"]:
        for n in range(1, max_streak+1):
            sub = df[(df["streak_dir"]==d)&(df["streak_len"]==n)]
            if len(sub) < 5: continue
            mr = 1 - sub["next_bull"].mean() if d=="bull" else sub["next_bull"].mean()
            results.append({
                "after":           f"{n}x {d}",
                "count":           len(sub),
                "next_bull_rate":  round(sub["next_bull"].mean(), 3),
                "revert_prob":     round(mr, 3),
            })
    return pd.DataFrame(results).sort_values("revert_prob", ascending=False)


# ─────────────────────────────────────────
#  PRINT HELPER
# ─────────────────────────────────────────
def sec(title, df, n=12):
    bar = "─" * 62
    print(f"\n┌{bar}┐")
    print(f"│  {title:<60}│")
    print(f"└{bar}┘")
    print(df.head(n).to_string(index=False))


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    print("\n" + "█"*64)
    print("  BTC/USDT FUTURES — FULL PATTERN MINER")
    print("  Jim Simons Method: Data First, No Thesis")
    print("█"*64)

    print("\n▶ Generating synthetic BTC data (mirrors Binance Futures)...")
    df_raw  = generate_btc_data(1000)
    funding = generate_funding_rates(df_raw["open_time"])
    print(f"  {len(df_raw)} hourly candles | BTC range: ${df_raw['close'].min():,.0f} – ${df_raw['close'].max():,.0f}")

    print("▶ Building features (CVD, VWAP, delta, volume z-score)...")
    df = build_features(df_raw)
    print(f"  {len(df)} usable rows\n")

    # ── Patterns 1–6 ──
    sec("PATTERN 1 — TIME OF DAY BIAS (top hours by edge)", pattern_time_of_day(df))
    sec("PATTERN 2 — DAY OF WEEK BIAS", pattern_day_of_week(df))
    sec("PATTERN 4 — ABOVE vs BELOW VWAP(20)", pattern_vwap(df))
    sec("PATTERN 5 — FUNDING RATE REGIME", pattern_funding(df, funding))
    sec("PATTERN 6 — CANDLE STREAK → REVERT PROBABILITY", pattern_streaks(df))

    # ── Pattern 7: Liquidation Clusters ──
    sec("PATTERN 7 — LIQUIDATION CLUSTER DETECTION (vol z-score)", pattern_liquidation_clusters(df))

    # ── Pattern 8: CVD Divergence ──
    print(f"\n┌{'─'*62}┐")
    print(f"│  {'PATTERN 8 — CVD DIVERGENCE SCORING':<60}│")
    print(f"└{'─'*62}┘")
    cvd = pattern_cvd_divergence(df)
    print("\n  Score table (all candles):")
    print(cvd.to_string(index=False))

    # ── Final Summary ──
    print(f"\n\n{'█'*64}")
    print("  EDGE SUMMARY — STRONGEST SIGNALS FOUND")
    print(f"{'█'*64}")

    tod = pattern_time_of_day(df)
    dow = pattern_day_of_week(df)
    strk = pattern_streaks(df)
    liq = pattern_liquidation_clusters(df)

    best_hour = tod.iloc[0]
    worst_hour = tod.iloc[1]
    print(f"\n  ● Best hour (UTC):   {int(best_hour['hour']):02d}:00  →  bull_rate={best_hour['bull_rate']:.1%}  edge={best_hour['edge']:.3f}")
    print(f"  ● 2nd best hour:     {int(worst_hour['hour']):02d}:00  →  bull_rate={worst_hour['bull_rate']:.1%}  edge={worst_hour['edge']:.3f}")

    best_day = dow.iloc[0]
    print(f"  ● Best day:          {best_day['day_of_week']}  →  bull_rate={best_day['bull_rate']:.1%}")

    best_streak = strk.iloc[0]
    print(f"  ● Best streak setup: {best_streak['after']}  →  revert_prob={best_streak['revert_prob']:.1%}  (n={best_streak['count']})")

    if not liq.empty:
        bl = liq.sort_values("avg_vol_zscore", ascending=False).iloc[0]
        print(f"  ● Strongest liq evt: {bl['event']}  →  next_bull={bl['next_bull_rate']:.1%}  (n={bl['count']})")

    cvd_score3 = cvd[cvd["div_score"] >= 2]
    if not cvd_score3.empty:
        c = cvd_score3.iloc[0]
        print(f"  ● CVD div score ≥2:  next_bull_rate={c['next_bull_rate']:.1%}  avg_ret={c['avg_next_ret_%']:.4f}%")

    print(f"\n  Candles analyzed:  {len(df)}")
    print(f"  Funding records:   {len(funding)}")
    print(f"  Patterns checked:  8")
    print(f"\n  → Take the top 2–3 edges. Combine them. That's your setup filter.")
    print(f"  → Replace synthetic data with live Binance API for real signals.\n")

if __name__ == "__main__":
    main()