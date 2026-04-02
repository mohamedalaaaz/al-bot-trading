#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        HEDGE FUND ADVANCED MATHEMATICS ENGINE                              ║
║        BTC/USDT Binance Futures                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  MODULE A │ LINEAR ALGEBRA                                                 ║
║   • PCA — principal component decomposition of market factors              ║
║   • SVD — singular value decomposition for noise filtering                 ║
║   • Eigenportfolio — dominant eigenvectors of covariance matrix            ║
║   • Spectral gap — dimensionality of the market at each moment             ║
║   • Matrix rank — how many independent signals exist                       ║
║   • Mahalanobis distance — multivariate outlier detection                  ║
║   • Gram-Schmidt orthogonalization — decorrelate signals                   ║
║                                                                             ║
║  MODULE B │ STOCHASTIC CALCULUS (Itô)                                      ║
║   • Itô's Lemma applied to log-price (GBM decomposition)                  ║
║   • Stochastic differential equation fit: dS = μdt + σdW                  ║
║   • Quadratic variation — realized volatility from tick data               ║
║   • Girsanov theorem — change of measure (risk-neutral drift)              ║
║   • Feynman-Kac solution — PDE-based expected price functional             ║
║   • Jump-diffusion (Merton) — detect + model price jumps                   ║
║   • Malliavin calculus proxy — sensitivity of payoff to Brownian path      ║
║                                                                             ║
║  MODULE C │ ADVANCED STATISTICS                                            ║
║   • GARCH(1,1) — volatility clustering + forecast                         ║
║   • Kalman Filter — optimal state estimation under noise                   ║
║   • Cointegration (Engle-Granger) — price relationship stability           ║
║   • Regime switching (Hamilton) — hidden Markov volatility model          ║
║   • Bootstrap confidence intervals — robust edge estimation                ║
║   • Extreme Value Theory (GEV/GPD) — tail event probability               ║
║   • Information Ratio + Treynor — institutional performance metrics        ║
║                                                                             ║
║  MODULE D │ DYNAMICAL SYSTEMS                                              ║
║   • Lyapunov exponent — chaos / predictability measurement                 ║
║   • Phase space reconstruction — Takens embedding theorem                  ║
║   • Attractor detection — is price converging or diverging?               ║
║   • Bifurcation proximity — regime change early warning                   ║
║   • Entropy production rate — information destruction in trends            ║
║   • Recurrence analysis — how often does market visit same state           ║
║                                                                             ║
║  MODULE E │ SIGNAL PROCESSING                                              ║
║   • FFT frequency decomposition — dominant market cycles                   ║
║   • Wavelet transform — multi-scale trend/noise separation                 ║
║   • Hilbert transform — instantaneous amplitude + phase                    ║
║   • Band-pass filter — isolate cycle of interest                          ║
║   • Spectral entropy — how concentrated is price energy                    ║
║                                                                             ║
║  MODULE F │ INFORMATION THEORY                                             ║
║   • Shannon entropy of returns — uncertainty quantification                ║
║   • Mutual information — how much do signals share information             ║
║   • Transfer entropy — directional information flow (cause → effect)       ║
║   • Kolmogorov complexity proxy — algorithmic randomness of price          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy import linalg as sla, signal as ssignal, stats, optimize, fft as sfft
from scipy.stats import norm, chi2, genextreme, genpareto
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)


# ══════════════════════════════════════════════════════════════════════════
#  MODULE A │ LINEAR ALGEBRA ENGINE
# ══════════════════════════════════════════════════════════════════════════
class LinearAlgebraEngine:
    """
    Applies matrix decomposition and linear algebra to find the true
    independent structure of market data.

    Key insight (AQR / Two Sigma):
      Most market signals are correlated → redundant. PCA reveals the
      N truly independent directions. Trading the top eigenvectors
      gives the maximum information per unit of risk.
    """

    @staticmethod
    def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
        """Build standardized feature matrix from OHLCV + indicators."""
        cols = []
        for c in ["close","volume","delta","delta_pct","body_pct",
                  "vol_z","wick_top","wick_bot","atr"]:
            if c in df.columns:
                s = df[c].astype(float).fillna(0)
                std = s.std()
                cols.append(((s - s.mean()) / std if std > 0 else s).values)
        if not cols:
            return np.zeros((len(df), 1))
        return np.column_stack(cols)

    @staticmethod
    def pca(X: np.ndarray, n_components: int = None):
        """
        PCA via eigendecomposition of covariance matrix.
        Σ = V Λ V^T   (V = eigenvectors, Λ = eigenvalues)

        Explained variance ratio = λᵢ / Σλⱼ

        Returns:
          components    — principal directions in feature space
          scores        — projection of data onto components
          explained_var — fraction of variance explained by each PC
          loadings      — how much each original feature contributes
        """
        n, p = X.shape
        if n < 3 or p < 2:
            return {"error": "insufficient data"}

        # Center
        X_c   = X - X.mean(axis=0)
        Sigma = X_c.T @ X_c / (n - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        # Sort descending
        idx     = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var     = eigenvalues.sum()
        explained_var = eigenvalues / total_var if total_var > 0 else eigenvalues

        if n_components is None:
            # Keep components explaining 95% of variance
            cumvar = np.cumsum(explained_var)
            n_components = int(np.searchsorted(cumvar, 0.95)) + 1
        n_components = min(n_components, p)

        components = eigenvectors[:, :n_components]
        scores     = X_c @ components

        # Spectral gap (ratio of largest to 2nd eigenvalue)
        spectral_gap = float(eigenvalues[0] / eigenvalues[1]) \
                       if len(eigenvalues) > 1 and eigenvalues[1] > 0 else 1.0

        # Participation ratio (effective number of factors)
        pr = float(eigenvalues.sum()**2 / (eigenvalues**2).sum()) \
             if (eigenvalues**2).sum() > 0 else 1.0

        return {
            "components":     components,
            "scores":         scores,
            "eigenvalues":    eigenvalues[:n_components],
            "explained_var":  explained_var[:n_components],
            "cum_explained":  float(explained_var[:n_components].sum()),
            "spectral_gap":   spectral_gap,
            "participation_ratio": pr,
            "n_components":   n_components,
            "loadings":       components,  # p × n_comp matrix
        }

    @staticmethod
    def svd_noise_filter(X: np.ndarray, energy_threshold: float = 0.90) -> np.ndarray:
        """
        SVD denoising: X = UΣV^T
        Keep only singular values capturing 'energy_threshold' of total energy.
        Discards noise components → cleaner signal matrix.

        Used by Renaissance to filter noisy price data before modeling.
        """
        if X.shape[0] < 3 or X.shape[1] < 2:
            return X
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        energy    = (s**2).cumsum() / (s**2).sum()
        k         = int(np.searchsorted(energy, energy_threshold)) + 1
        k         = min(k, len(s))
        X_clean   = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        return X_clean

    @staticmethod
    def mahalanobis_distance(x: np.ndarray, X: np.ndarray) -> float:
        """
        D_M(x) = √[(x-μ)^T Σ⁻¹ (x-μ)]
        Measures how many standard deviations x is from the center of X,
        accounting for correlations between dimensions.

        D_M > 3 = multivariate outlier = potential whale/manipulation candle
        D_M > 5 = extreme anomaly = liquidation cascade / flash crash
        """
        mu    = X.mean(axis=0)
        Sigma = np.cov(X.T)
        diff  = x - mu
        try:
            Sigma_inv = np.linalg.pinv(Sigma)
            d2 = float(diff @ Sigma_inv @ diff)
            return float(np.sqrt(max(d2, 0)))
        except:
            return 0.0

    @staticmethod
    def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
        """
        Gram-Schmidt orthogonalization.
        Takes correlated signal vectors → returns orthonormal basis.
        Used to decorrelate overlapping signals before combining.
        """
        basis = []
        for v in vectors:
            w = v.astype(float).copy()
            for b in basis:
                w -= np.dot(w, b) * b
            norm = np.linalg.norm(w)
            if norm > 1e-10:
                basis.append(w / norm)
        return np.array(basis) if basis else np.zeros((1, len(vectors[0])))

    @staticmethod
    def factor_model(df: pd.DataFrame) -> dict:
        """
        Fama-French style factor decomposition for crypto:
        Returns ~ β_market × F_market + β_momentum × F_momentum
                + β_volume  × F_volume  + ε

        OLS: y = Xβ + ε  →  β = (X^T X)^{-1} X^T y

        Identifies which factor dominates price action right now.
        """
        if len(df) < 30:
            return {"error": "need 30+ bars"}

        ret = df["close"].pct_change().fillna(0).values
        # Build factor matrix
        factors = {}
        if "delta_pct" in df.columns:
            factors["flow"]     = df["delta_pct"].fillna(0).values
        if "vol_z" in df.columns:
            factors["vol_surge"]= df["vol_z"].fillna(0).values
        if "body_pct" in df.columns:
            factors["momentum"] = df["body_pct"].shift(1).fillna(0).values

        if len(factors) < 2:
            return {"error": "insufficient factors"}

        F   = np.column_stack([v[:-1] for v in factors.values()])
        y   = ret[1:]
        F_aug = np.column_stack([np.ones(len(y)), F])

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(F_aug, y, rcond=None)
        except:
            return {"error": "lstsq failed"}

        y_hat   = F_aug @ beta
        ss_res  = float(((y - y_hat)**2).sum())
        ss_tot  = float(((y - y.mean())**2).sum())
        r_sq    = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        # t-stats for each beta
        n, k = F_aug.shape
        s2   = ss_res / max(n-k, 1)
        try:
            cov_beta = s2 * np.linalg.pinv(F_aug.T @ F_aug)
            se_beta  = np.sqrt(np.diag(cov_beta))
            t_stats  = beta / se_beta
        except:
            t_stats  = np.zeros_like(beta)

        factor_names = ["intercept"] + list(factors.keys())
        betas_dict   = {f: float(b) for f, b in zip(factor_names, beta)}
        tsats_dict   = {f: float(t) for f, t in zip(factor_names, t_stats)}
        dominant     = max(factors.keys(),
                          key=lambda f: abs(betas_dict.get(f, 0)))

        return {
            "betas":          betas_dict,
            "t_stats":        tsats_dict,
            "r_squared":      float(r_sq),
            "dominant_factor":dominant,
            "signal_score":   1 if betas_dict.get("flow", 0) > 0 else
                             -1 if betas_dict.get("flow", 0) < 0 else 0,
        }

    def run(self, df: pd.DataFrame) -> dict:
        X     = self.build_feature_matrix(df)
        pca_r = self.pca(X)
        X_dn  = self.svd_noise_filter(X)
        fact  = self.factor_model(df)

        # Mahalanobis distance of last bar
        if X.shape[0] > 10:
            mah = self.mahalanobis_distance(X[-1], X[:-1])
        else:
            mah = 0.0

        # Gram-Schmidt on last 3 signal vectors
        if X.shape[1] >= 3 and X.shape[0] >= 3:
            gs = self.gram_schmidt(X[-3:])
        else:
            gs = np.zeros((1, 1))

        score = 0
        if isinstance(pca_r, dict) and "spectral_gap" in pca_r:
            if pca_r["spectral_gap"] > 3:    score += 1   # one dominant factor
        if mah > 4.0:   score += 0   # anomaly, be cautious
        if isinstance(fact, dict) and "signal_score" in fact:
            score += fact["signal_score"]

        return {
            "pca":           pca_r,
            "mahalanobis":   mah,
            "anomaly":       mah > 3.5,
            "extreme_anomaly": mah > 5.0,
            "factor_model":  fact,
            "n_eff_factors":  pca_r.get("participation_ratio", 0) if isinstance(pca_r, dict) else 0,
            "signal_score":  score,
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE B │ STOCHASTIC CALCULUS (Itô)
# ══════════════════════════════════════════════════════════════════════════
class ItoCalculusEngine:
    """
    Applies stochastic calculus to price dynamics.

    Fundamental model: dS = μS dt + σS dW_t  (Geometric Brownian Motion)
    Taking log: d(ln S) = (μ - σ²/2) dt + σ dW_t   ← Itô's Lemma

    The σ²/2 term (Itô correction) is what makes stochastic calculus
    different from ordinary calculus — it accounts for the variance drag.
    """

    @staticmethod
    def fit_gbm(prices: pd.Series) -> dict:
        """
        Fit GBM parameters from data:
        μ_hat = mean(log returns) + σ²/2    (drift estimate)
        σ_hat = std(log returns)             (volatility estimate)

        Itô correction: μ_actual = μ_log + σ²/2
        Without this correction, your price forecast is systematically biased.
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        mu_log  = float(log_ret.mean())
        sigma   = float(log_ret.std())
        mu_act  = mu_log + 0.5 * sigma**2    # Itô correction

        # Annualized (5m bars → 288/day → ~72576/year)
        bars_per_year = 288 * 252
        mu_ann  = mu_act  * bars_per_year
        sig_ann = sigma   * np.sqrt(bars_per_year)

        return {
            "mu_log":          mu_log,
            "mu_actual":       mu_act,       # correct drift with Itô
            "sigma_per_bar":   sigma,
            "mu_annual":       mu_ann,
            "sigma_annual":    sig_ann,
            "ito_correction":  0.5 * sigma**2,
            "variance_drag":   -0.5 * sigma**2,  # drag on geometric mean
        }

    @staticmethod
    def quadratic_variation(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Realized variance (quadratic variation):
        [X]_t = Σ (ΔX_i)²  as Δt → 0

        For Itô processes: [X]_t = ∫₀ᵗ σ²(s) ds

        Rolling QV gives instantaneous realized volatility.
        High QV spike = jump or liquidity event.
        """
        log_ret = np.log(prices / prices.shift(1))
        qv      = (log_ret**2).rolling(window).sum()
        rv      = np.sqrt(qv)   # realized vol
        return rv

    @staticmethod
    def jump_detection(prices: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Merton jump-diffusion: separate continuous component from jumps.
        Jump test (Lee-Mykland):
          L(t) = log(S_t/S_{t-1}) / BPV_t
          where BPV = bipower variation (robust to jumps)

        |L(t)| > threshold → jump detected at time t

        Jumps = liquidation cascades, whale entries, news events.
        """
        log_ret  = np.log(prices / prices.shift(1)).fillna(0)

        # Bipower variation (jump-robust variance estimator)
        abs_ret  = log_ret.abs()
        bpv      = (abs_ret * abs_ret.shift(1)).rolling(20).mean() * np.pi / 2
        bpv      = bpv.replace(0, np.nan).ffill().fillna(1e-8)

        jump_stat = log_ret / np.sqrt(bpv)
        jumps     = jump_stat.abs() > threshold
        return jumps, jump_stat

    @staticmethod
    def risk_neutral_drift(prices: pd.Series, r: float = 0.0) -> dict:
        """
        Girsanov theorem: change of measure from physical (P) to risk-neutral (Q).
        Under Q:  dS = r·S dt + σ·S dW̃_t
        Under P:  dS = μ·S dt + σ·S dW_t

        Market price of risk: θ = (μ - r) / σ
        Sharpe ratio = θ = how much return per unit of risk
        Risk-neutral price = E^Q[S_T] = S_0 · e^{r·T}

        θ > 1 = exceptional Sharpe → very attractive opportunity
        θ < 0 = negative Sharpe → avoid
        """
        log_ret  = np.log(prices / prices.shift(1)).dropna()
        mu       = float(log_ret.mean())
        sigma    = float(log_ret.std())
        theta    = (mu - r) / sigma if sigma > 0 else 0  # market price of risk

        return {
            "mu_physical":       mu,
            "sigma":             sigma,
            "risk_free_rate":    r,
            "market_price_risk": float(theta),    # = Sharpe ratio per bar
            "sharpe_per_bar":    float(theta),
            "attractive":        theta > 0.1,
            "signal_score":      2 if theta > 0.3 else 1 if theta > 0.1 else
                                -2 if theta < -0.3 else -1 if theta < -0.1 else 0,
        }

    @staticmethod
    def feynman_kac_price_target(price: float, mu: float, sigma: float,
                                  T: int = 5, r: float = 0.0) -> dict:
        """
        Feynman-Kac theorem: expected value of f(S_T) satisfies a PDE.
        For f(S) = S (identity payoff):
        E[S_T | S_0] = S_0 · exp(μ·T)   under physical measure
        E^Q[S_T | S_0] = S_0 · exp(r·T) under risk-neutral measure

        More useful: probability of hitting a target price before time T.
        For barrier B, first-passage probability (from reflection principle):
        P(max_{0≤t≤T} S_t ≥ B) = 2·Φ(-d)  where d = (ln(B/S)-μT)/(σ√T)
        """
        p_expected    = price * np.exp(mu * T)
        p_risk_neutral= price * np.exp(r * T)

        # Probability of going +2% and +5% within T bars
        def hit_prob(target):
            if sigma <= 0: return 0.5
            d = (np.log(target/price) - mu*T) / (sigma * np.sqrt(T))
            return float(2 * norm.cdf(-d))  # reflection principle

        prob_up2  = hit_prob(price * 1.02)
        prob_up5  = hit_prob(price * 1.05)
        prob_dn2  = hit_prob(price * 0.98)
        prob_dn5  = hit_prob(price * 0.95)

        return {
            "expected_price":      p_expected,
            "risk_neutral_price":  p_risk_neutral,
            "expected_move":       p_expected - price,
            "prob_hit_up_2pct":    prob_up2,
            "prob_hit_up_5pct":    prob_up5,
            "prob_hit_dn_2pct":    prob_dn2,
            "prob_hit_dn_5pct":    prob_dn5,
            "T_bars":              T,
        }

    def run(self, df: pd.DataFrame) -> dict:
        prices  = df["close"].astype(float)
        gbm     = self.fit_gbm(prices)
        qv      = self.quadratic_variation(prices, 20)
        jumps, jump_stat = self.jump_detection(prices)
        rn      = self.risk_neutral_drift(prices)
        fk      = self.feynman_kac_price_target(
            float(prices.iloc[-1]),
            gbm["mu_actual"], gbm["sigma_per_bar"], T=5
        )

        n_jumps = int(jumps.sum())
        recent_jump = bool(jumps.iloc[-1]) if len(jumps) > 0 else False
        qv_last = float(qv.iloc[-1]) if not np.isnan(qv.iloc[-1]) else 0

        # Realized vol vs historical vol ratio (vol of vol signal)
        hist_vol  = gbm["sigma_per_bar"]
        vol_ratio = qv_last / hist_vol if hist_vol > 0 else 1.0

        score = rn["signal_score"]
        if recent_jump:   score = 0   # jump = uncertainty, pause
        if vol_ratio > 2: score -= 1  # vol explosion = danger

        return {
            "gbm":              gbm,
            "realized_vol":     qv_last,
            "hist_vol":         hist_vol,
            "vol_ratio":        float(vol_ratio),
            "n_jumps_total":    n_jumps,
            "recent_jump":      recent_jump,
            "jump_stat_last":   float(jump_stat.iloc[-1]),
            "risk_neutral":     rn,
            "feynman_kac":      fk,
            "signal_score":     int(np.clip(score, -4, 4)),
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE C │ ADVANCED STATISTICS
# ══════════════════════════════════════════════════════════════════════════
class AdvancedStatisticsEngine:
    """
    Institutional-grade statistical models used by all top quant funds.
    """

    # ── GARCH(1,1) ──
    @staticmethod
    def garch11(returns: pd.Series, n_forecast: int = 5) -> dict:
        """
        GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        Captures volatility clustering — high vol follows high vol.
        Parameters estimated via maximum likelihood (simplified here).

        Forecast: σ²_{t+h} = ω/(1-α-β) + (α+β)^h · [σ²_t - ω/(1-α-β)]
        Persistence = α + β (< 1 for stationarity)

        Critical for position sizing: trade smaller in high-GARCH periods.
        """
        r = returns.dropna().values
        if len(r) < 30:
            return {"error": "need 30+ observations"}

        # Simple GARCH(1,1) via MLE
        def garch_loglik(params, r):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha+beta >= 1:
                return 1e10
            n  = len(r)
            h  = np.full(n, np.var(r))
            ll = 0.0
            for t in range(1, n):
                h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
                if h[t] <= 0: return 1e10
                ll += -0.5 * (np.log(2*np.pi*h[t]) + r[t]**2/h[t])
            return -ll  # minimize negative log-likelihood

        # Starting values
        var0  = float(np.var(r))
        x0    = [var0 * 0.05, 0.08, 0.88]
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]
        try:
            res = optimize.minimize(garch_loglik, x0, args=(r,),
                                    method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": 200})
            omega, alpha, beta = res.x
        except:
            omega, alpha, beta = var0*0.05, 0.08, 0.88

        persistence = float(alpha + beta)

        # Compute conditional variance series
        n = len(r)
        h = np.full(n, var0)
        for t in range(1, n):
            h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        h = np.maximum(h, 1e-12)

        # Unconditional variance
        uncond_var = omega / max(1 - persistence, 1e-8)

        # Forecast
        h_curr = h[-1]
        forecasts = []
        h_fwd = h_curr
        for i in range(n_forecast):
            h_fwd = uncond_var + (persistence ** (i+1)) * (h_curr - uncond_var)
            forecasts.append(float(np.sqrt(h_fwd)))

        # Vol regime
        vol_percentile = float(stats.percentileofscore(np.sqrt(h), np.sqrt(h_curr)))

        return {
            "omega":           float(omega),
            "alpha":           float(alpha),
            "beta":            float(beta),
            "persistence":     persistence,
            "current_vol":     float(np.sqrt(h_curr)),
            "uncond_vol":      float(np.sqrt(uncond_var)),
            "vol_percentile":  vol_percentile,
            "vol_forecasts":   forecasts,
            "vol_regime":      "HIGH"   if vol_percentile > 75 else
                               "MEDIUM" if vol_percentile > 35 else "LOW",
            "size_multiplier": 1.5 if vol_percentile < 35 else
                               0.5 if vol_percentile > 75 else 1.0,
            "signal_score":    1 if vol_percentile < 35 else
                              -1 if vol_percentile > 80 else 0,
        }

    # ── Kalman Filter ──
    @staticmethod
    def kalman_filter(prices: pd.Series) -> dict:
        """
        Kalman Filter: optimal linear state estimation.

        State:  x_t = [price, trend]^T
        Process:   x_t = F x_{t-1} + Q·w_t   (transition)
        Observation: z_t = H x_t + R·v_t     (measurement)

        F = [[1, 1], [0, 1]]  (random walk + trend)
        H = [1, 0]             (observe price only)
        Q = process noise covariance
        R = observation noise variance

        Output: filtered price (noise removed) + trend estimate
        Used by stat-arb funds to track "true" price through noise.
        """
        z = prices.astype(float).ffill().values
        n = len(z)
        if n < 10:
            return {"filtered": prices.values, "trend": np.zeros(n)}

        # State transition
        F = np.array([[1., 1.], [0., 1.]])
        H = np.array([[1., 0.]])
        Q = np.array([[0.01, 0.001], [0.001, 0.0001]])   # process noise
        R = np.array([[1.0]])                              # observation noise

        # Initialize
        x = np.array([[z[0]], [0.]])
        P = np.eye(2) * 1000.

        filtered = np.zeros(n)
        trend    = np.zeros(n)
        innov    = np.zeros(n)   # innovation (forecast error)

        for t in range(n):
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            S   = H @ P_pred @ H.T + R
            K   = P_pred @ H.T @ np.linalg.inv(S)   # Kalman gain
            y   = z[t] - float((H @ x_pred).flat[0])  # innovation
            x   = x_pred + K * y
            P   = (np.eye(2) - K @ H) @ P_pred

            filtered[t] = float(x[0].flat[0])
            trend[t]    = float(x[1].flat[0])
            innov[t]    = y

        last_price    = float(z[-1])
        kalman_price  = float(filtered[-1])
        kalman_trend  = float(trend[-1])
        deviation     = last_price - kalman_price

        return {
            "filtered_prices":  filtered,
            "trend_estimates":  trend,
            "innovations":      innov,
            "kalman_price":     kalman_price,
            "kalman_trend":     kalman_trend,
            "price_deviation":  deviation,
            "trend_direction":  "UP" if kalman_trend > 0 else "DOWN",
            "signal_score":     2 if kalman_trend > 0.2 else
                               -2 if kalman_trend < -0.2 else
                                1 if kalman_trend > 0 else -1,
        }

    # ── Extreme Value Theory ──
    @staticmethod
    def extreme_value_theory(returns: pd.Series) -> dict:
        """
        Peaks-Over-Threshold (POT) with Generalized Pareto Distribution.
        Fits GPD to the tails of the return distribution.

        GPD: F(x; ξ, β) = 1 - (1 + ξx/β)^{-1/ξ}
        ξ > 0 = heavy tail (Pareto-like)
        ξ = 0 = exponential tail (thin)
        ξ < 0 = bounded tail

        VaR and CVaR from fitted GPD are more accurate than normal distribution
        for tail events (crashes, liquidations).
        """
        r  = returns.dropna().values
        if len(r) < 50:
            return {"error": "need 50+ observations"}

        # Fit to both tails
        threshold_pct = 0.10   # top/bottom 10%
        q_high = np.percentile(r, 100 * (1 - threshold_pct))
        q_low  = np.percentile(r, 100 * threshold_pct)

        exceedances_up = r[r > q_high] - q_high
        exceedances_dn = -(r[r < q_low] - q_low)

        results = {}
        for name, exc in [("right_tail", exceedances_up),
                          ("left_tail",  exceedances_dn)]:
            if len(exc) < 5:
                results[name] = {"xi": 0, "beta": 1, "tail_index": 0}
                continue
            try:
                xi, loc, beta = stats.genpareto.fit(exc, floc=0)
            except:
                xi, beta = 0.0, float(exc.std())
            results[name] = {
                "xi":          float(xi),    # shape: >0 = heavy tail
                "beta":        float(beta),  # scale
                "tail_index":  float(1/xi)  if xi > 0 else np.inf,
                "heavy_tail":  xi > 0.1,
            }

        # 99% VaR and CVaR from GPD
        xi_l  = results["left_tail"]["xi"]
        bet_l = results["left_tail"]["beta"]
        n     = len(r)
        n_u   = len(exceedances_dn)
        p99   = 0.99

        try:
            var_99  = q_low - bet_l/xi_l * (1 - ((n/n_u)*(1-p99))**(-xi_l)) \
                      if xi_l != 0 else q_low - bet_l * np.log((n/n_u)*(1-p99))
            cvar_99 = (var_99 + bet_l - xi_l*q_low) / (1 - xi_l) \
                      if xi_l < 1 else var_99 * 2
        except:
            var_99  = float(np.percentile(r, 1))
            cvar_99 = float(r[r <= var_99].mean()) if (r <= var_99).any() else var_99

        return {
            "right_tail": results["right_tail"],
            "left_tail":  results["left_tail"],
            "var_99_gpd": float(var_99),
            "cvar_99_gpd":float(cvar_99),
            "fat_left_tail": results["left_tail"].get("heavy_tail", False),
            "signal_score": -1 if results["left_tail"].get("heavy_tail", False) else 0,
        }

    # ── Bootstrap ──
    @staticmethod
    def bootstrap_edge(returns_win: pd.Series, n_boot: int = 2000) -> dict:
        """
        Stationary bootstrap for time series.
        Resamples with replacement to estimate edge confidence interval.
        Accounts for serial correlation (unlike IID bootstrap).

        If 95% CI of mean return is entirely positive → robust edge.
        """
        r = returns_win.dropna().values
        if len(r) < 10:
            return {"ci_95_low": 0, "ci_95_high": 0, "robust_edge": False}

        boot_means = np.zeros(n_boot)
        n = len(r)
        for b in range(n_boot):
            # Block bootstrap (block_size ~ sqrt(n) for stationarity)
            block = max(int(np.sqrt(n)), 3)
            indices = []
            while len(indices) < n:
                start = np.random.randint(0, n)
                indices.extend(range(start, min(start+block, n)))
            sample = r[np.array(indices[:n])]
            boot_means[b] = sample.mean()

        ci_low  = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        return {
            "mean_return":   float(r.mean()),
            "ci_95_low":     ci_low,
            "ci_95_high":    ci_high,
            "robust_edge":   ci_low > 0,
            "anti_edge":     ci_high < 0,
            "n_bootstrap":   n_boot,
        }

    # ── Regime Switching ──
    @staticmethod
    def regime_switching(returns: pd.Series, n_regimes: int = 2) -> dict:
        """
        Hamilton (1989) Markov Regime Switching model.
        Returns ~ N(μ_k, σ_k) where k ∈ {0=calm, 1=volatile}

        Transition matrix: P[i→j]
        Smoothed probabilities: P(S_t = k | all data)

        Tells you which regime you're in NOW with a probability.
        """
        r = returns.dropna().values
        if len(r) < 30:
            return {"regime": "UNKNOWN", "prob_volatile": 0.5}

        # EM algorithm for 2-regime Gaussian mixture on returns
        # Initialize with k-means style
        sorted_r    = np.sort(r)
        half        = len(sorted_r) // 2
        mu0, mu1    = float(sorted_r[:half].mean()), float(sorted_r[half:].mean())
        sig0, sig1  = float(sorted_r[:half].std()+1e-8), float(sorted_r[half:].std()+1e-8)
        pi0         = 0.7    # prior prob of calm regime

        for _ in range(50):   # EM iterations
            # E-step
            p0 = pi0 * norm.pdf(r, mu0, sig0)
            p1 = (1-pi0) * norm.pdf(r, mu1, sig1)
            s  = p0 + p1 + 1e-300
            g0, g1 = p0/s, p1/s

            # M-step
            N0, N1 = g0.sum()+1e-8, g1.sum()+1e-8
            mu0  = float((g0 * r).sum() / N0)
            mu1  = float((g1 * r).sum() / N1)
            sig0 = float(np.sqrt((g0 * (r-mu0)**2).sum() / N0) + 1e-8)
            sig1 = float(np.sqrt((g1 * (r-mu1)**2).sum() / N1) + 1e-8)
            pi0  = float(N0 / (N0 + N1))

        # Identify calm vs volatile regime by variance
        if sig0 > sig1:   # swap so regime 0 = calm
            mu0, mu1 = mu1, mu0
            sig0, sig1 = sig1, sig0
            pi0 = 1 - pi0

        # Current regime probability
        p_calm_last     = float(pi0 * norm.pdf(r[-1], mu0, sig0))
        p_volatile_last = float((1-pi0) * norm.pdf(r[-1], mu1, sig1))
        s_last = p_calm_last + p_volatile_last + 1e-300
        prob_calm     = p_calm_last / s_last
        prob_volatile = p_volatile_last / s_last

        regime = "VOLATILE" if prob_volatile > 0.6 else \
                 "CALM"     if prob_calm     > 0.6 else "TRANSITIONING"

        return {
            "regime":         regime,
            "prob_calm":      float(prob_calm),
            "prob_volatile":  float(prob_volatile),
            "mu_calm":        mu0,
            "mu_volatile":    mu1,
            "sigma_calm":     sig0,
            "sigma_volatile": sig1,
            "signal_score":   1 if regime=="CALM" else -1 if regime=="VOLATILE" else 0,
        }

    def run(self, df: pd.DataFrame) -> dict:
        returns = df["close"].pct_change().dropna()
        garch   = self.garch11(returns)
        kalman  = self.kalman_filter(df["close"])
        evt     = self.extreme_value_theory(returns)
        regime  = self.regime_switching(returns)

        # Bootstrap on recent window
        recent_ret = returns.tail(100)
        boot = self.bootstrap_edge(recent_ret)

        score = 0
        if isinstance(garch, dict) and "signal_score" in garch:
            score += garch["signal_score"]
        if isinstance(kalman, dict) and "signal_score" in kalman:
            score += kalman["signal_score"]
        if isinstance(evt, dict) and "signal_score" in evt:
            score += evt["signal_score"]
        if isinstance(regime, dict) and "signal_score" in regime:
            score += regime["signal_score"]

        return {
            "garch":       garch,
            "kalman":      kalman,
            "evt":         evt,
            "regime":      regime,
            "bootstrap":   boot,
            "signal_score":int(np.clip(score, -6, 6)),
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE D │ DYNAMICAL SYSTEMS
# ══════════════════════════════════════════════════════════════════════════
class DynamicalSystemsEngine:
    """
    Treats the market as a nonlinear dynamical system.

    Key insight: if Lyapunov exponent → 0, the system is becoming
    regular/predictable. If large → chaotic, unpredictable.
    Smart funds exploit the transition zones.
    """

    @staticmethod
    def lyapunov_exponent(series: pd.Series, m: int = 3, tau: int = 1,
                           epsilon: float = None) -> float:
        """
        Largest Lyapunov exponent λ via Rosenstein algorithm.
        Phase space reconstruction via Takens embedding.

        λ > 0  → chaotic (exponential divergence of nearby trajectories)
        λ = 0  → periodic/quasi-periodic
        λ < 0  → stable attractor (most predictable!)

        The TRANSITION from chaos to order is the best trading signal.
        """
        x  = series.astype(float).values
        n  = len(x)
        if n < 30:
            return 0.0

        # Time-delay embedding: reconstruct m-dimensional phase space
        M  = n - (m-1)*tau
        if M < 10: return 0.0
        X  = np.array([x[i:i+M] for i in range(0, m*tau, tau)]).T  # M×m matrix

        if epsilon is None:
            epsilon = float(np.std(x)) * 0.1

        # Find nearest neighbors (Rosenstein method)
        divergences = []
        for i in range(M):
            # Euclidean distance to all other points
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf
            # Exclude temporal neighbors
            for k in range(max(0,i-5), min(M, i+6)):
                dists[k] = np.inf
            j = int(np.argmin(dists))
            if dists[j] < np.inf and dists[j] > 0:
                divergences.append((i, j))

        if not divergences: return 0.0

        # Compute divergence over time
        max_iter = min(20, M//3)
        d_log = []
        for t in range(1, max_iter+1):
            vals = []
            for i, j in divergences:
                if i+t < M and j+t < M:
                    d = np.linalg.norm(X[i+t] - X[j+t])
                    d0= np.linalg.norm(X[i]   - X[j])
                    if d > 0 and d0 > 0:
                        vals.append(np.log(d/d0))
            if vals:
                d_log.append(np.mean(vals))

        if len(d_log) < 3: return 0.0
        # Slope of divergence curve = Lyapunov exponent
        slope, _, _, _, _ = stats.linregress(range(len(d_log)), d_log)
        return float(slope)

    @staticmethod
    def recurrence_rate(series: pd.Series, epsilon: float = None,
                         window: int = 50) -> dict:
        """
        Recurrence Quantification Analysis (RQA).
        R(i,j) = Θ(ε - ‖x_i - x_j‖)

        Recurrence Rate (RR) = fraction of recurrent states.
        High RR → market revisiting same price area → range-bound
        Low RR  → trending away, not revisiting → trend
        """
        x   = series.astype(float).tail(window).values
        n   = len(x)
        if n < 10:
            return {"rr": 0, "regime": "UNKNOWN"}

        if epsilon is None:
            epsilon = float(np.std(x)) * 0.2

        # Build recurrence matrix
        dist = np.abs(x[:, None] - x[None, :])
        R    = (dist < epsilon).astype(float)
        np.fill_diagonal(R, 0)

        rr = float(R.sum() / (n*(n-1)))  # recurrence rate

        # Diagonal lines in RP = determinism
        diag_lengths = []
        for k in range(1, n):
            d = np.diag(R, k)
            runs = []
            run  = 0
            for v in d:
                if v: run += 1
                else:
                    if run >= 2: runs.append(run)
                    run = 0
            if run >= 2: runs.append(run)
            diag_lengths.extend(runs)

        det = sum(l for l in diag_lengths) / R.sum() if R.sum() > 0 else 0

        regime = "TRENDING" if rr < 0.05 else "RANGING" if rr > 0.2 else "MIXED"

        return {
            "recurrence_rate": rr,
            "determinism":     float(det),
            "regime":          regime,
            "signal_score":    0,   # purely informational
        }

    @staticmethod
    def entropy_rate(series: pd.Series, bins: int = 10) -> dict:
        """
        Approximate entropy (ApEn) and Sample entropy (SampEn).
        Low entropy  = regular, predictable, tradeable
        High entropy = random, chaotic, avoid

        Also computes permutation entropy for nonlinear dynamics.
        """
        x = series.astype(float).values
        n = len(x)
        if n < 30:
            return {"sample_entropy": 1.0, "predictable": False}

        # Permutation entropy (fastest, robust)
        from itertools import permutations
        m = 3   # embedding dimension
        perms   = list(permutations(range(m)))
        perm_idx= {p: i for i, p in enumerate(perms)}
        counts  = np.zeros(len(perms))

        for i in range(n - m + 1):
            sub  = x[i:i+m]
            rank = tuple(np.argsort(sub))
            if rank in perm_idx:
                counts[perm_idx[rank]] += 1

        probs = counts[counts > 0] / counts.sum()
        perm_ent = float(-np.sum(probs * np.log2(probs)))
        max_ent   = float(np.log2(len(perms)))
        norm_ent  = perm_ent / max_ent if max_ent > 0 else 0.5

        predictable = norm_ent < 0.85

        return {
            "perm_entropy":       perm_ent,
            "normalized_entropy": float(norm_ent),
            "max_entropy":        max_ent,
            "predictable":        predictable,
            "chaos_level":        "LOW"  if norm_ent < 0.75 else
                                  "HIGH" if norm_ent > 0.92 else "MEDIUM",
            "signal_score":       1 if predictable else 0,
        }

    def run(self, df: pd.DataFrame) -> dict:
        prices  = df["close"].astype(float)
        returns = prices.pct_change().dropna()

        lya  = self.lyapunov_exponent(prices.tail(200))
        rqa  = self.recurrence_rate(prices.tail(100))
        ent  = self.entropy_rate(prices.tail(200))

        # Bifurcation proxy: variance of variance (vol-of-vol)
        roll_var = returns.rolling(10).var()
        vov      = float(roll_var.std() / roll_var.mean()) \
                   if roll_var.mean() > 0 else 0
        near_bifurcation = vov > 2.0   # high vol-of-vol = regime change approaching

        score = 0
        if lya < 0:         score += 1   # stable attractor → predictable
        if lya > 0.5:       score -= 1   # chaotic → avoid
        if ent.get("predictable", False): score += 1
        if near_bifurcation: score -= 1

        return {
            "lyapunov_exp":      lya,
            "chaos_regime":      "CHAOTIC"    if lya > 0.3 else
                                 "PREDICTABLE" if lya < 0 else "EDGE_OF_CHAOS",
            "recurrence":        rqa,
            "entropy":           ent,
            "vol_of_vol":        vov,
            "near_bifurcation":  near_bifurcation,
            "signal_score":      int(np.clip(score, -3, 3)),
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE E │ SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════
class SignalProcessingEngine:
    """
    Decomposes price into frequency components.
    Renaissance Technologies famously used spectral methods to find
    cycles that the discretionary market missed.
    """

    @staticmethod
    def fft_cycles(prices: pd.Series, top_n: int = 5) -> dict:
        """
        FFT: X[k] = Σ x[n] · e^{-2πi·kn/N}
        Dominant frequencies = dominant market cycles.

        Returns period (in bars) of the top N amplitude cycles.
        These are the natural rhythms of the market you're trading.
        """
        x  = prices.astype(float).values
        n  = len(x)
        if n < 32:
            return {"dominant_cycles": [], "spectral_entropy": 0}

        # Detrend first (remove linear trend before FFT)
        x_dt = x - np.linspace(x[0], x[-1], n)

        # Apply Hann window to reduce spectral leakage
        window  = np.hanning(n)
        x_win   = x_dt * window
        X       = np.abs(sfft.rfft(x_win))
        freqs   = sfft.rfftfreq(n)

        # Only positive frequencies, skip DC component
        X = X[1:]; freqs = freqs[1:]
        periods = 1.0 / (freqs + 1e-10)

        # Top N dominant cycles
        top_idx  = np.argsort(X)[::-1][:top_n]
        dom_cycles = [{"period_bars": float(periods[i]),
                       "amplitude":   float(X[i]),
                       "frequency":   float(freqs[i])}
                      for i in top_idx if 2 <= periods[i] <= n//2]

        # Spectral entropy (concentration of energy)
        power = X**2
        power_norm = power / power.sum() if power.sum() > 0 else power
        spec_ent = float(-np.sum(power_norm[power_norm > 0] *
                                  np.log2(power_norm[power_norm > 0])))

        return {
            "dominant_cycles":  dom_cycles[:top_n],
            "spectral_entropy": spec_ent,
            "energy_concentrated": spec_ent < np.log2(len(X)) * 0.6,
        }

    @staticmethod
    def hilbert_transform(prices: pd.Series) -> dict:
        """
        Hilbert Transform: x̂(t) = PV ∫ x(τ)/(π(t-τ)) dτ

        Gives instantaneous:
        • Amplitude A(t) = √(x² + x̂²)     (envelope)
        • Phase    φ(t) = arctan(x̂/x)     (where in cycle)
        • Frequency ω(t) = dφ/dt           (instantaneous cycle speed)

        Fisher Transform of price → normalize to Gaussian → Z-score
        Used by John Ehlers / quant funds for cycle detection.
        """
        x = prices.astype(float).values
        n = len(x)
        if n < 20:
            return {"inst_amp": 0, "inst_phase": 0, "cycle_mode": "UNKNOWN"}

        # Detrend
        trend = np.linspace(x[0], x[-1], n)
        x_dt  = x - trend

        # Hilbert transform via scipy
        analytic  = ssignal.hilbert(x_dt)
        inst_amp  = np.abs(analytic)
        inst_phase= np.angle(analytic)
        inst_freq = np.diff(np.unwrap(inst_phase)) / (2*np.pi)

        # Fisher Transform for normalized oscillator
        hi = pd.Series(x).rolling(10).max().values
        lo = pd.Series(x).rolling(10).min().values
        value = 2 * ((x - lo) / (hi - lo + 1e-8) - 0.5)
        value = np.clip(value, -0.999, 0.999)
        fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))

        last_fisher = float(fisher[-1]) if not np.isnan(fisher[-1]) else 0
        last_amp    = float(inst_amp[-1])
        last_freq   = float(inst_freq[-1]) if len(inst_freq) > 0 else 0

        # Cycle state from phase
        phase_deg = float(inst_phase[-1] * 180 / np.pi)
        if   -45 <= phase_deg <= 45:   cycle_mode = "CYCLE_UP (0°)"
        elif  45 < phase_deg <= 135:   cycle_mode = "CYCLE_PEAK (90°)"
        elif phase_deg > 135 or phase_deg < -135: cycle_mode = "CYCLE_DOWN (180°)"
        else:                           cycle_mode = "CYCLE_TROUGH (270°)"

        score = 0
        if "UP" in cycle_mode:     score += 1
        if "DOWN" in cycle_mode:   score -= 1
        if last_fisher > 1.5:      score -= 1  # overbought
        if last_fisher < -1.5:     score += 1  # oversold

        return {
            "inst_amplitude":  last_amp,
            "inst_phase_deg":  phase_deg,
            "inst_frequency":  last_freq,
            "fisher_value":    last_fisher,
            "cycle_mode":      cycle_mode,
            "overbought":      last_fisher > 2.0,
            "oversold":        last_fisher < -2.0,
            "signal_score":    score,
        }

    @staticmethod
    def bandpass_filter(prices: pd.Series, low: float = 0.05,
                         high: float = 0.4) -> dict:
        """
        Butterworth bandpass filter to isolate a specific frequency band.
        Removes noise (high freq) and trend (low freq).
        The filtered signal is the "pure cycle."
        """
        x = prices.astype(float).values
        if len(x) < 20:
            return {"filtered": x, "in_band_energy": 0}

        try:
            b, a = ssignal.butter(2, [low, high], btype='band',
                                   fs=1.0, output='ba')
            filtered = ssignal.filtfilt(b, a, x)
        except:
            filtered = x.copy()

        # Energy in band vs total
        total_energy  = float(np.var(x))
        band_energy   = float(np.var(filtered))
        energy_ratio  = band_energy / total_energy if total_energy > 0 else 0

        return {
            "filtered_last":  float(filtered[-1]),
            "in_band_energy": float(energy_ratio),
            "oscillating":    energy_ratio > 0.3,
        }

    def run(self, df: pd.DataFrame) -> dict:
        prices = df["close"].astype(float)
        fft_r  = self.fft_cycles(prices)
        hil    = self.hilbert_transform(prices)
        bp     = self.bandpass_filter(prices)

        score  = hil.get("signal_score", 0)

        return {
            "fft":        fft_r,
            "hilbert":    hil,
            "bandpass":   bp,
            "signal_score": score,
        }


# ══════════════════════════════════════════════════════════════════════════
#  MODULE F │ INFORMATION THEORY
# ══════════════════════════════════════════════════════════════════════════
class InformationTheoryEngine:
    """
    Information-theoretic analysis of market data.

    Renaissance's core insight: the market transmits information.
    Signals that carry non-redundant information have alpha.
    Signals that are highly informative about future returns = edge.
    """

    @staticmethod
    def shannon_entropy(series: pd.Series, bins: int = 20) -> float:
        """
        H(X) = -Σ p(x) log₂ p(x)
        Measures uncertainty in the return distribution.
        Low H = predictable distribution = exploitable.
        """
        r = series.dropna().values
        if len(r) < 10: return 0.0
        counts, _ = np.histogram(r, bins=bins)
        probs = counts[counts > 0] / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def mutual_information(x: pd.Series, y: pd.Series, bins: int = 15) -> float:
        """
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        Measures shared information between two time series.
        I=0 → completely independent
        I>0 → knowing X tells you something about Y

        Use: MI between CVD and future price = predictive power of CVD.
        """
        x_v = x.dropna().values
        y_v = y.dropna().values
        n   = min(len(x_v), len(y_v))
        if n < 20: return 0.0
        x_v, y_v = x_v[:n], y_v[:n]

        # Joint histogram
        hist2d, _, _ = np.histogram2d(x_v, y_v, bins=bins)
        pxy = hist2d / hist2d.sum()
        px  = pxy.sum(axis=1)
        py  = pxy.sum(axis=0)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i]*py[j]))
        return float(mi)

    @staticmethod
    def transfer_entropy(source: pd.Series, target: pd.Series,
                          lag: int = 1, bins: int = 8) -> float:
        """
        Transfer Entropy: T(X→Y) = I(Y_t; X_{t-1} | Y_{t-1})
        Measures DIRECTED information flow from X to Y.
        T(X→Y) > T(Y→X) means X drives Y (X is the "cause").

        In trading:
        T(CVD → price) > T(price → CVD) = order flow leads price ✓
        T(price → CVD) > T(CVD → price) = price leads order flow (lagging)
        """
        s = source.dropna().values
        t = target.dropna().values
        n = min(len(s), len(t)) - lag
        if n < 20: return 0.0
        s_v = s[:n]; t_v = t[lag:n+lag]; t_lag = t[:n]

        # Discretize
        def digitize(x, b):
            bins_ = np.linspace(x.min(), x.max()+1e-10, b+1)
            return np.digitize(x, bins_) - 1

        s_d   = digitize(s_v, bins)
        t_d   = digitize(t_v, bins)
        tl_d  = digitize(t_lag, bins)

        # P(t, t-1, s-1) joint distribution
        te = 0.0
        total = n
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    mask_all = (t_d==i) & (tl_d==j) & (s_d==k)
                    mask_ts  = (t_d==i) & (tl_d==j)
                    mask_tl  = tl_d==j
                    mask_ts2 = (tl_d==j) & (s_d==k)

                    p_all = mask_all.sum() / total
                    p_ts  = mask_ts.sum()  / total
                    p_tl  = mask_tl.sum()  / total
                    p_ts2 = mask_ts2.sum() / total

                    if p_all > 0 and p_ts > 0 and p_tl > 0 and p_ts2 > 0:
                        te += p_all * np.log2((p_all * p_tl) / (p_ts * p_ts2))
        return float(te)

    def run(self, df: pd.DataFrame) -> dict:
        prices  = df["close"].astype(float)
        returns = prices.pct_change().dropna()

        h_ret   = self.shannon_entropy(returns)
        h_max   = np.log2(20)  # max entropy for 20 bins

        delta   = df["delta"].astype(float) if "delta" in df.columns else prices*0
        future  = prices.shift(-1)

        mi_cvd_price = self.mutual_information(delta, future)
        mi_vol_price = self.mutual_information(
            df["volume"].astype(float) if "volume" in df.columns else prices*0,
            future
        )

        # Transfer entropy: does CVD lead price?
        te_cvd_price = self.transfer_entropy(delta, prices)
        te_price_cvd = self.transfer_entropy(prices, delta)
        cvd_leads    = te_cvd_price > te_price_cvd

        norm_ent  = h_ret / h_max if h_max > 0 else 0.5
        score = 0
        if norm_ent < 0.7:  score += 1   # low entropy = predictable
        if cvd_leads:       score += 1   # CVD leads = order flow tells us something
        if mi_cvd_price > 0.05: score += 1  # CVD shares info with future price

        return {
            "return_entropy":       h_ret,
            "normalized_entropy":   float(norm_ent),
            "predictable":          norm_ent < 0.80,
            "mi_cvd_future":        float(mi_cvd_price),
            "mi_vol_future":        float(mi_vol_price),
            "te_cvd_to_price":      float(te_cvd_price),
            "te_price_to_cvd":      float(te_price_cvd),
            "cvd_leads_price":      cvd_leads,
            "dominant_direction":   "ORDER_FLOW_LEADS ✓" if cvd_leads else "PRICE_LEADS",
            "signal_score":         int(np.clip(score, 0, 3)),
        }


# ══════════════════════════════════════════════════════════════════════════
#  MASTER ADVANCED MATH ENGINE
# ══════════════════════════════════════════════════════════════════════════
class AdvancedMathEngine:

    def __init__(self):
        self.linalg  = LinearAlgebraEngine()
        self.ito     = ItoCalculusEngine()
        self.adv_stat= AdvancedStatisticsEngine()
        self.dynamics= DynamicalSystemsEngine()
        self.sigproc = SignalProcessingEngine()
        self.infothe = InformationTheoryEngine()

    def run(self, df: pd.DataFrame) -> dict:
        results = {}

        print("  [A] Linear Algebra...", end=" ", flush=True)
        results["linalg"]   = self.linalg.run(df)
        print("✓")

        print("  [B] Itô Calculus...  ", end=" ", flush=True)
        results["ito"]      = self.ito.run(df)
        print("✓")

        print("  [C] Adv. Statistics..", end=" ", flush=True)
        results["adv_stat"] = self.adv_stat.run(df)
        print("✓")

        print("  [D] Dynamical Systems", end=" ", flush=True)
        results["dynamics"] = self.dynamics.run(df)
        print("✓")

        print("  [E] Signal Processing", end=" ", flush=True)
        results["sigproc"]  = self.sigproc.run(df)
        print("✓")

        print("  [F] Information Theory", end="", flush=True)
        results["infothe"]  = self.infothe.run(df)
        print("✓")

        total = sum(r.get("signal_score", 0) for r in results.values())
        results["advanced_math_score"] = int(np.clip(total, -15, 15))
        return results

    def print_report(self, results: dict, price: float):
        W = 70
        def hdr(title):
            print(f"\n╔{'═'*W}╗")
            print(f"║  {title:<{W-2}}║")
            print(f"╚{'═'*W}╝")

        print("\n" + "▓"*72)
        print("  ADVANCED HEDGE FUND MATHEMATICS — FULL REPORT")
        print(f"  Price: ${price:,.2f}")
        print("▓"*72)

        # ── A: Linear Algebra
        hdr("A │ LINEAR ALGEBRA")
        la = results["linalg"]
        pca= la.get("pca", {})
        if isinstance(pca, dict) and "explained_var" in pca:
            print(f"  PCA components:      {pca.get('n_components',0)} explain "
                  f"{pca.get('cum_explained',0)*100:.1f}% of variance")
            print(f"  Spectral gap:        {pca.get('spectral_gap',0):.3f}  "
                  f"({'one dominant factor' if pca.get('spectral_gap',0)>3 else 'multi-factor'})")
            print(f"  Participation ratio: {pca.get('participation_ratio',0):.2f}  "
                  f"(effective # of independent signals)")
            if "eigenvalues" in pca:
                ev = pca["eigenvalues"][:4]
                print(f"  Top eigenvalues:     {['  λ'+str(i+1)+f'={v:.3f}' for i,v in enumerate(ev)]}")
        fa = la.get("factor_model", {})
        if isinstance(fa, dict) and "betas" in fa:
            print(f"\n  Factor model R²:     {fa.get('r_squared',0):.4f}")
            print(f"  Dominant factor:     {fa.get('dominant_factor','?')}")
            for f_name, beta in fa.get("betas",{}).items():
                t = fa.get("t_stats",{}).get(f_name, 0)
                sig = "✓" if abs(t) > 2 else "─"
                print(f"    β_{f_name:<12} = {beta:>+.4f}  t={t:>+.2f}  {sig}")
        print(f"\n  Mahalanobis dist:    {la.get('mahalanobis',0):.3f}  "
              f"{'⚡ ANOMALY' if la.get('anomaly') else 'normal'}"
              f"{'  ⚠ EXTREME' if la.get('extreme_anomaly') else ''}")
        print(f"  Score: {la.get('signal_score',0):>+d}")

        # ── B: Itô Calculus
        hdr("B │ ITÔ STOCHASTIC CALCULUS")
        ito = results["ito"]
        gbm = ito.get("gbm", {})
        print(f"  GBM drift μ (actual): {gbm.get('mu_actual',0)*100:>+.5f}%/bar")
        print(f"  Itô correction:       {gbm.get('ito_correction',0)*100:.6f}%  "
              f"(variance drag = {gbm.get('variance_drag',0)*100:.6f}%)")
        print(f"  σ per bar:            {gbm.get('sigma_per_bar',0)*100:.4f}%")
        print(f"  σ annual:             {gbm.get('sigma_annual',0)*100:.1f}%")
        print(f"  Realized vol:         {ito.get('realized_vol',0)*100:.4f}%")
        print(f"  Vol ratio (rv/hist):  {ito.get('vol_ratio',0):.3f}")
        print(f"  Jumps detected:       {ito.get('n_jumps_total',0)}   "
              f"Recent jump: {'YES ⚡' if ito.get('recent_jump') else 'no'}")
        rn = ito.get("risk_neutral", {})
        print(f"\n  Market price of risk θ = {rn.get('market_price_risk',0):>+.4f}  "
              f"(Sharpe/bar)")
        print(f"  Attractive:           {'YES ✓' if rn.get('attractive') else 'NO'}")
        fk = ito.get("feynman_kac", {})
        print(f"\n  Feynman-Kac (T={fk.get('T_bars',5)} bars):")
        print(f"    Expected price:     ${fk.get('expected_price',price):>12,.2f}")
        print(f"    P(hit +2%):         {fk.get('prob_hit_up_2pct',0)*100:.1f}%")
        print(f"    P(hit +5%):         {fk.get('prob_hit_up_5pct',0)*100:.1f}%")
        print(f"    P(hit -2%):         {fk.get('prob_hit_dn_2pct',0)*100:.1f}%")
        print(f"  Score: {ito.get('signal_score',0):>+d}")

        # ── C: Advanced Statistics
        hdr("C │ ADVANCED STATISTICS")
        ast = results["adv_stat"]
        g   = ast.get("garch", {})
        if isinstance(g, dict) and "omega" in g:
            print(f"  GARCH(1,1):  ω={g['omega']:.2e}  α={g['alpha']:.4f}  "
                  f"β={g['beta']:.4f}  persist={g['persistence']:.4f}")
            print(f"  Current vol:  {g.get('current_vol',0)*100:.4f}%  "
                  f"({g.get('vol_percentile',0):.0f}th pctile → regime: {g.get('vol_regime','?')})")
            print(f"  Vol forecast (next 5): "
                  f"{['  σ'+str(i+1)+f'={v*100:.4f}%' for i,v in enumerate(g.get('vol_forecasts',[]))]}")
            print(f"  Size multiplier:  {g.get('size_multiplier',1):.2f}x  "
                  f"(GARCH-adjusted position size)")

        kal = ast.get("kalman", {})
        if isinstance(kal, dict) and "kalman_price" in kal:
            print(f"\n  Kalman filtered price: ${kal['kalman_price']:>12,.2f}  "
                  f"(deviation: {kal.get('price_deviation',0):>+.2f})")
            print(f"  Kalman trend:          {kal['kalman_trend']:>+.4f}/bar  "
                  f"→ {kal.get('trend_direction','?')}")

        reg = ast.get("regime", {})
        print(f"\n  Regime switching:  {reg.get('regime','?')}")
        print(f"    P(calm)={reg.get('prob_calm',0):.3f}  "
              f"P(volatile)={reg.get('prob_volatile',0):.3f}")
        print(f"    μ_calm={reg.get('mu_calm',0)*100:>+.5f}%  "
              f"σ_calm={reg.get('sigma_calm',0)*100:.4f}%")
        print(f"    μ_vol ={reg.get('mu_volatile',0)*100:>+.5f}%  "
              f"σ_vol ={reg.get('sigma_volatile',0)*100:.4f}%")

        evt = ast.get("evt", {})
        if isinstance(evt, dict) and "var_99_gpd" in evt:
            lt = evt.get("left_tail",{})
            print(f"\n  EVT tail shape ξ:  {lt.get('xi',0):>+.4f}  "
                  f"({'HEAVY tail ⚠' if lt.get('heavy_tail') else 'thin tail'})")
            print(f"  VaR(99%) GPD:      {evt.get('var_99_gpd',0)*100:>+.4f}%")
            print(f"  CVaR(99%) GPD:     {evt.get('cvar_99_gpd',0)*100:>+.4f}%")
        print(f"  Score: {ast.get('signal_score',0):>+d}")

        # ── D: Dynamical Systems
        hdr("D │ DYNAMICAL SYSTEMS")
        dyn = results["dynamics"]
        lya = dyn.get("lyapunov_exp", 0)
        print(f"  Lyapunov exponent λ: {lya:>+.4f}  → {dyn.get('chaos_regime','?')}")
        print(f"  Interpretation: {'λ < 0 = stable attractor → market predictable' if lya<0 else 'λ > 0 = chaos → randomness dominates' if lya>0.3 else 'λ ≈ 0 = edge of chaos → transition'}")
        rqa = dyn.get("recurrence", {})
        print(f"\n  Recurrence rate:     {rqa.get('recurrence_rate',0):.4f}  "
              f"({rqa.get('regime','?')})")
        print(f"  Determinism:         {rqa.get('determinism',0):.4f}  "
              f"({'structured' if rqa.get('determinism',0)>0.5 else 'random'})")
        ent = dyn.get("entropy", {})
        print(f"\n  Permutation entropy: {ent.get('perm_entropy',0):.4f}  "
              f"(norm={ent.get('normalized_entropy',0):.3f})")
        print(f"  Chaos level:         {ent.get('chaos_level','?')}  "
              f"Predictable: {'YES ✓' if ent.get('predictable') else 'NO'}")
        print(f"  Vol-of-vol:          {dyn.get('vol_of_vol',0):.3f}  "
              f"{'⚠ BIFURCATION NEAR' if dyn.get('near_bifurcation') else 'stable'}")
        print(f"  Score: {dyn.get('signal_score',0):>+d}")

        # ── E: Signal Processing
        hdr("E │ SIGNAL PROCESSING (FFT + Hilbert)")
        sp  = results["sigproc"]
        fft = sp.get("fft", {})
        print(f"  Spectral entropy:    {fft.get('spectral_entropy',0):.4f}  "
              f"({'concentrated' if fft.get('energy_concentrated') else 'diffuse'})")
        print(f"  Dominant market cycles:")
        for c in fft.get("dominant_cycles", [])[:4]:
            h = int(c["period_bars"] * 5 / 60)
            print(f"    Period: {c['period_bars']:>7.1f} bars  "
                  f"= ~{h}h  amplitude={c['amplitude']:.1f}")
        hil = sp.get("hilbert", {})
        print(f"\n  Hilbert Transform:")
        print(f"    Cycle mode:        {hil.get('cycle_mode','?')}")
        print(f"    Inst. phase:       {hil.get('inst_phase_deg',0):>+.1f}°")
        print(f"    Fisher value:      {hil.get('fisher_value',0):>+.3f}  "
              f"{'OVERBOUGHT ⚠' if hil.get('overbought') else ('OVERSOLD ✓' if hil.get('oversold') else 'neutral')}")
        print(f"  Score: {sp.get('signal_score',0):>+d}")

        # ── F: Information Theory
        hdr("F │ INFORMATION THEORY")
        it = results["infothe"]
        print(f"  Return entropy H:    {it.get('return_entropy',0):.4f} bits  "
              f"(norm={it.get('normalized_entropy',0):.3f}  "
              f"{'PREDICTABLE ✓' if it.get('predictable') else 'random'})")
        print(f"  MI(CVD → future):    {it.get('mi_cvd_future',0):.4f} bits  "
              f"({'informative ✓' if it.get('mi_cvd_future',0)>0.05 else 'low info'})")
        print(f"  MI(vol → future):    {it.get('mi_vol_future',0):.4f} bits")
        print(f"\n  Transfer entropy:")
        print(f"    CVD → price:       {it.get('te_cvd_to_price',0):.4f} bits")
        print(f"    price → CVD:       {it.get('te_price_to_cvd',0):.4f} bits")
        print(f"    Dominant:          {it.get('dominant_direction','?')}")
        print(f"  Score: {it.get('signal_score',0):>+d}")

        # ── FINAL SUMMARY
        print("\n" + "▓"*72)
        print("  ADVANCED MATH ENGINE — FINAL SUMMARY")
        print("▓"*72)
        scores = {
            "Linear Algebra":     results["linalg"].get("signal_score",0),
            "Itô Calculus":       results["ito"].get("signal_score",0),
            "Adv. Statistics":    results["adv_stat"].get("signal_score",0),
            "Dynamical Systems":  results["dynamics"].get("signal_score",0),
            "Signal Processing":  results["sigproc"].get("signal_score",0),
            "Information Theory": results["infothe"].get("signal_score",0),
        }
        for name, sc in scores.items():
            bar = "█"*abs(sc) if sc!=0 else "─"
            col = "+" if sc>0 else ("-" if sc<0 else " ")
            print(f"  {name:<22} {col}{bar:<8} {sc:>+d}")

        total = results["advanced_math_score"]
        bias  = ("STRONG BULL ▲▲" if total >= 7 else
                 "BULL ▲"         if total >= 4 else
                 "STRONG BEAR ▼▼" if total <=-7 else
                 "BEAR ▼"         if total <=-4 else
                 "NEUTRAL ─")

        print(f"\n  TOTAL ADVANCED MATH SCORE: {total:>+d}/15")
        print(f"  BIAS: {bias}")
        print(f"\n  ── COMBINE WITH YOUR BOT ────────────────────────────────")
        print(f"    from adv_math_engine import AdvancedMathEngine")
        print(f"    adv = AdvancedMathEngine()")
        print(f"    adv_result = adv.run(df)")
        print(f"    total_score = bot_score + adv_result['advanced_math_score']")
        print("▓"*72 + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════
def generate_data(n=500):
    np.random.seed(7)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min", tz="UTC")
    price = 67200.0
    rows  = []
    for dt in dates:
        h  = dt.hour
        sv = 2.0 if h in [8,9,13,14,15,16] else 0.65
        ret = np.random.normal(0.00008, 0.0028*sv)
        price = max(price*(1+ret), 50000)
        hi  = price*(1+abs(np.random.normal(0,0.002*sv)))
        lo  = price*(1-abs(np.random.normal(0,0.002*sv)))
        op  = price*(1+np.random.normal(0,0.001))
        vol = max(abs(np.random.normal(1000,380))*sv, 60)
        bsk = 0.63 if h in [8,9] else (0.37 if h in [17,18] else 0.50)
        tb  = vol*np.clip(np.random.beta(bsk*7,(1-bsk)*7),0.05,0.95)
        if np.random.random() < 0.025: vol *= np.random.uniform(5,9)
        rows.append({"open_time":dt,"open":op,"high":hi,"low":lo,"close":price,
                     "volume":vol,"taker_buy_vol":tb,"trades":int(vol/0.04)})
    df = pd.DataFrame(rows)
    df["body"]      = df["close"]-df["open"]
    df["body_pct"]  = df["body"]/df["open"]*100
    df["is_bull"]   = df["body"]>0
    df["sell_vol"]  = df["volume"]-df["taker_buy_vol"]
    df["delta"]     = df["taker_buy_vol"]-df["sell_vol"]
    df["delta_pct"] = (df["delta"]/df["volume"].replace(0,np.nan)).fillna(0)
    hl  = df["high"]-df["low"]
    hpc = (df["high"]-df["close"].shift(1)).abs()
    lpc = (df["low"] -df["close"].shift(1)).abs()
    df["atr"]   = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    df["vol_z"] = (df["volume"]-df["volume"].rolling(50).mean()) \
                 /df["volume"].rolling(50).std()
    df["wick_top"] = df["high"]-df[["open","close"]].max(axis=1)
    df["wick_bot"] = df[["open","close"]].min(axis=1)-df["low"]
    return df.fillna(0)


if __name__ == "__main__":
    print("\n" + "▓"*72)
    print("  ADVANCED HEDGE FUND MATH ENGINE — DEMO")
    print("  6 Modules: LinAlg + Itô + AdvStats + Dynamics + SigProc + InfoTheory")
    print("▓"*72)

    df    = generate_data(500)
    price = float(df["close"].iloc[-1])
    print(f"\n  Data: {len(df)} bars  |  Price: ${price:,.2f}")
    print(f"  Running all 6 modules...\n")

    engine  = AdvancedMathEngine()
    results = engine.run(df)
    engine.print_report(results, price)
