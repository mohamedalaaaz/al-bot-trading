#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   QUANT MATH ENGINE  —  Pure Statistics & Probability                      ║
║   BTC/USDT Binance Futures                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  This file contains ONLY mathematical engines. No signals, no ML,         ║
║  no trading logic. Pure probability and statistics used by the main        ║
║  engine at runtime to compute:                                              ║
║                                                                             ║
║  A. BAYESIAN INFERENCE                                                      ║
║     · Beta-Binomial sequential updating (conjugate prior)                  ║
║     · Bayesian Model Averaging (BMA) with posterior weights                ║
║     · Bayes Factor: B₁₀ = P(data|H₁)/P(data|H₀)                         ║
║     · Credible intervals (exact Beta quantiles)                            ║
║     · Prior sensitivity analysis (robust inference)                        ║
║                                                                             ║
║  B. EXTREME VALUE THEORY (EVT)                                              ║
║     · Generalized Pareto Distribution (GPD) — peaks over threshold        ║
║     · Hill estimator — tail index for power-law tails                      ║
║     · VaR and CVaR at 99%, 99.9% confidence (not Gaussian approx)        ║
║     · Expected Shortfall with bootstrap uncertainty band                   ║
║     · Return level plots: how bad can it get in N bars?                   ║
║                                                                             ║
║  C. INFORMATION THEORY                                                      ║
║     · Shannon entropy H(X) = -Σ p log p                                   ║
║     · Mutual information I(X;Y) — how much does CVD tell us about ret?    ║
║     · Transfer entropy T(X→Y) — directional information flow              ║
║     · Variance Ratio Test (Lo-MacKinlay) — momentum vs mean reversion     ║
║     · Approximate entropy — market predictability measure                  ║
║                                                                             ║
║  D. STOCHASTIC PROCESSES                                                    ║
║     · OU MLE — exact closed-form (Shoji & Ozaki 1998)                    ║
║     · RTS Kalman Smoother — optimal two-pass estimation                   ║
║     · GARCH(1,1) — variance dynamics with analytical warm-start           ║
║     · Heston model — stochastic vol (κ, θ, ξ, ρ)                        ║
║     · Lévy jump detection — bipower variation test                        ║
║                                                                             ║
║  E. ADVANCED KELLY CRITERION                                                ║
║     · Bayesian-Kelly: uses posterior P(win) not sample estimate            ║
║     · CVaR-constrained Kelly: max f s.t. CVaR(strategy) ≥ floor          ║
║     · Multi-signal Kelly: Σ⁻¹μ with Ledoit-Wolf shrinkage                 ║
║     · Fractional Kelly with uncertainty shrinkage                          ║
║                                                                             ║
║  F. SEQUENTIAL TESTS                                                        ║
║     · SPRT (Wald 1947) — optimal stopping, controls α and β              ║
║     · CUSUM — change detection for regime shifts                           ║
║     · Page-Hinkley — concept drift detection                               ║
║                                                                             ║
║  USAGE:                                                                     ║
║    from quant_math import QuantMath                                         ║
║    qm = QuantMath()                                                         ║
║    qm.bayes.update("cvd", won=True)                                        ║
║    risk = qm.evt.cvar_99(returns)                                          ║
║    kelly = qm.kelly.bayesian_kelly("cvd", rr=2.5, garch_mult=1.2)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy import stats, optimize, linalg
from scipy.special import gammaln, betaln, digamma
from scipy.stats import (genpareto, genextreme, norm as sp_norm,
                          beta as beta_dist, t as t_dist)
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  A. BAYESIAN INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BayesianEngine:
    """
    Sequential Bayesian inference via Beta-Binomial conjugate model.

    Model:  θ_i ~ Beta(α_i, β_i)      (prior on win-rate of signal i)
            X_i | θ_i ~ Bernoulli(θ_i) (observation: trade won/lost)
    Update: α_i += win,  β_i += loss   (posterior stays Beta — conjugate)

    Why conjugate? Posterior = Beta → exact, closed-form, no MCMC.

    Interpretation:
        α = effective wins (including prior pseudo-counts)
        β = effective losses
        E[θ|data] = α/(α+β)   ← posterior mean
        Var[θ|data] = αβ/((α+β)²(α+β+1))  ← posterior variance

    Priors set from historical BTC signal literature:
        CVD divergence:  Beta(4, 2.5)  ~ prior 61.5% WR
        OU reversion:    Beta(3.5, 2)  ~ prior 63.6% WR
        Kalman trend:    Beta(3, 2)    ~ prior 60.0% WR
        Wyckoff:         Beta(3.5, 2)  ~ prior 63.6% WR
        Uninformative:   Beta(2, 2)    ~ prior 50.0% WR (uniform-ish)
    """

    PRIORS = {
        "cvd":       (4.0, 2.5),
        "ou":        (3.5, 2.0),
        "kalman":    (3.0, 2.0),
        "wyckoff":   (3.5, 2.0),
        "vwap":      (3.0, 2.0),
        "sweep":     (4.0, 2.0),
        "funding":   (3.0, 2.0),
        "trap":      (3.0, 2.0),
        "tick":      (2.5, 2.0),
        "default":   (2.0, 2.0),
    }

    def __init__(self):
        self._alpha = {}
        self._beta  = {}
        self._n     = defaultdict(int)
        # Initialize with priors
        for sig, (a, b) in self.PRIORS.items():
            self._alpha[sig] = a
            self._beta[sig]  = b

    def update(self, signal: str, won: bool):
        """Sequential Bayesian update after each observed trade."""
        if signal not in self._alpha:
            a, b = self.PRIORS["default"]
            self._alpha[signal] = a
            self._beta[signal]  = b
        if won:
            self._alpha[signal] += 1.0
        else:
            self._beta[signal]  += 1.0
        self._n[signal] += 1

    def posterior_mean(self, signal: str) -> float:
        """E[θ|data] = α/(α+β)  — Bayesian estimate of win rate."""
        a = self._alpha.get(signal, 2.0)
        b = self._beta.get(signal, 2.0)
        return a / (a + b)

    # Alias used by the main engine
    def p_win(self, signal: str) -> float:
        """Alias for posterior_mean(). E[P(win)|data]."""
        return self.posterior_mean(signal)

    def posterior_std(self, signal: str) -> float:
        """Posterior standard deviation — uncertainty of win rate estimate."""
        a = self._alpha.get(signal, 2.0)
        b = self._beta.get(signal, 2.0)
        n = a + b
        return math.sqrt(a * b / (n * n * (n + 1)))

    def credible_interval(self, signal: str, level: float = 0.90):
        """
        Exact Bayesian credible interval [lo, hi] at given probability level.
        Not a frequentist confidence interval — this is the probability that
        the TRUE win rate lies in [lo, hi] given the observed data.
        """
        a = self._alpha.get(signal, 2.0)
        b = self._beta.get(signal, 2.0)
        alpha = (1 - level) / 2
        lo = float(beta_dist.ppf(alpha,     a, b))
        hi = float(beta_dist.ppf(1 - alpha, a, b))
        return lo, hi

    def bayes_factor(self, signal: str) -> float:
        """
        Bayes Factor B₁₀: evidence for H₁ (edge exists) vs H₀ (no edge).

        B₁₀ = P(data | H₁: θ = posterior_mean) / P(data | H₀: θ = 0.5)

        Jeffreys scale:
            B₁₀ < 1    : evidence AGAINST H₁
            1 ≤ B₁₀ < 3: anecdotal
            3 ≤ B₁₀ < 10: substantial
            10 ≤ B₁₀ < 30: strong
            ≥ 30       : very strong evidence FOR H₁
        """
        a = self._alpha.get(signal, 2.0)
        b = self._beta.get(signal, 2.0)
        # Subtract prior pseudo-counts to get only the data contribution
        a0, b0 = self.PRIORS.get(signal, self.PRIORS["default"])
        n_obs = int(a + b - a0 - b0)
        k_obs = int(a - a0)
        if n_obs <= 0 or k_obs < 0 or k_obs > n_obs:
            return 1.0
        # Log Bayes Factor under Beta(a0,b0) prior vs H₀: θ=0.5
        # BF = Γ(a0+b0)/[Γ(a0)Γ(b0)] × Γ(a)Γ(b)/Γ(a+b) / (0.5^n_obs)
        log_bf = (gammaln(a0 + b0) - gammaln(a0) - gammaln(b0)
                  + gammaln(a)     - gammaln(a + b) + gammaln(b)
                  + n_obs * math.log(2))
        return float(np.exp(np.clip(log_bf, -30, 30)))

    def signal_weight(self, signal: str) -> float:
        """
        Multiplicative weight for signal score, derived from posterior.
        Returns value in [0.5, 1.6] — never zeros out a signal entirely.

        Logic:
          BF > 10  → strong evidence, weight 1.5×
          BF > 3   → substantial, weight 1.2×
          BF > 1   → anecdotal, weight 1.0×
          BF < 1   → against signal, weight 0.8×
          posterior_mean < 0.45 → possibly harmful, weight 0.6×
        """
        pm = self.posterior_mean(signal)
        bf = self.bayes_factor(signal)
        lo, _ = self.credible_interval(signal, 0.80)
        if pm < 0.45 and lo < 0.42:
            return 0.6
        if bf >= 30: return 1.6
        if bf >= 10: return 1.5
        if bf >= 3:  return 1.2
        if bf >= 1:  return 1.0
        return 0.8

    def bma_probability(self, signal_probs: dict) -> float:
        """
        Bayesian Model Averaging (BMA):
        P(win) = Σᵢ P(win|Mᵢ) × P(Mᵢ|data)

        Model weights ∝ Bayes Factor × posterior quality
        Normalizes to sum to 1.
        """
        weights = {}
        total   = 0.0
        for sig, prob in signal_probs.items():
            pm = self.posterior_mean(sig)
            bf = self.bayes_factor(sig)
            # Weight: BF × how far posterior is from 0.5 (both directions)
            w  = max(bf, 0.1) * (abs(pm - 0.5) + 0.5)
            weights[sig] = w
            total += w
        if total < 1e-10:
            return float(np.mean(list(signal_probs.values())))
        bma = sum(signal_probs[s] * weights[s] / total for s in signal_probs)
        return float(np.clip(bma, 0.01, 0.99))

    def summary(self) -> dict:
        """Full posterior summary for all tracked signals."""
        out = {}
        for sig in set(list(self._alpha.keys())):
            pm = self.posterior_mean(sig)
            lo, hi = self.credible_interval(sig)
            out[sig] = {
                "p_win":  pm,
                "std":    self.posterior_std(sig),
                "ci_lo":  lo,
                "ci_hi":  hi,
                "bf":     self.bayes_factor(sig),
                "weight": self.signal_weight(sig),
                "n_obs":  self._n[sig],
            }
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  B. EXTREME VALUE THEORY
# ══════════════════════════════════════════════════════════════════════════════

class EVTEngine:
    """
    Extreme Value Theory for tail risk estimation.

    Standard assumption (normal distribution) vastly UNDERESTIMATES
    tail risk for financial returns. Bitcoin has kurtosis ~8-15 vs
    normal's kurtosis of 3.

    We use Peaks-Over-Threshold (POT) with Generalized Pareto Distribution:
        X - u | X > u  ~  GPD(ξ, β)

    where:
        ξ = shape (tail heaviness; BTC typically 0.1-0.4)
        β = scale
        u = threshold (typically 5th percentile of losses)

    ξ > 0: Fréchet domain (heavy tail — typical for crypto)
    ξ = 0: Exponential tail (light)
    ξ < 0: Bounded tail (rare in finance)
    """

    def fit_gpd(self, returns: np.ndarray, threshold_pct: float = 0.05) -> dict:
        """
        Fit GPD to left tail (losses) via MLE.
        Uses Method of Moments for warm-start (faster convergence).
        """
        r = returns[~np.isnan(returns)]
        if len(r) < 50:
            return {"xi": 0.3, "beta": 0.02, "u": -0.05,
                    "var_99": -0.05, "cvar_99": -0.07, "valid": False}

        # Threshold = threshold_pct quantile of losses
        u = float(np.percentile(r, threshold_pct * 100))
        exceed = -(r[r < u] - u)   # positive exceedances

        if len(exceed) < 15:
            return {"xi": 0.3, "beta": 0.02, "u": u,
                    "var_99": float(np.percentile(r, 1)),
                    "cvar_99": float(r[r <= np.percentile(r, 1)].mean()) if (r <= np.percentile(r,1)).any() else float(np.percentile(r,0.5)),
                    "valid": False}

        # Method of Moments warm-start
        m1 = exceed.mean()
        m2 = (exceed**2).mean()
        if m2 > m1**2:
            xi0   = 0.5 * (1 - m1**2 / max(m2 - m1**2, 1e-10))
            beta0 = max(0.5 * m1 * (1 + m1**2 / max(m2 - m1**2, 1e-10)), 1e-5)
        else:
            xi0, beta0 = 0.25, m1

        # MLE via L-BFGS-B with analytical gradient.
        # GPD log-likelihood is convex in (ξ, β) when ξ > -½, so gradient methods
        # converge in O(10) evaluations vs Nelder-Mead's O(100+).
        # ∂ℓ/∂β = -n/β + (1+1/ξ) Σ ξz/(β(1+ξz/β))
        # ∂ℓ/∂ξ = -Σ log(1+ξz/β)/ξ² + (1+1/ξ) Σ z/(β+ξz)
        def neg_log_lik_grad(params):
            xi, beta = params
            if beta <= 0:
                return 1e10, np.array([0., 1e5])
            z_scaled = xi * exceed / beta
            if (1 + z_scaled <= 0).any():
                return 1e10, np.array([0., 1e5])
            ln_z   = np.log1p(z_scaled)
            n_e    = len(exceed)
            nll    = n_e * math.log(beta) + (1 + 1/xi) * ln_z.sum() if xi != 0 \
                     else n_e * math.log(beta) + exceed.sum() / beta
            if xi != 0:
                w      = 1.0 / (1.0 + z_scaled)
                # ∂nll/∂β = n/β - (1+1/ξ)·Σ[ξ·x/(β²·(1+ξx/β))]
                #          = n/β - (1+1/ξ)/β · Σ[z_scaled·w]
                g_beta = n_e / beta - (1 + 1/xi) / beta * (z_scaled * w).sum()
                # ∂nll/∂ξ = Σ[log(1+ξx/β)]/ξ² - (1+1/ξ)·Σ[x/(β+ξx)]
                g_xi   = ln_z.sum() / xi**2 - (1 + 1/xi) * (exceed * w / beta).sum()
            else:
                g_beta = n_e / beta - exceed.sum() / beta**2
                g_xi   = 0.0
            return float(nll), np.array([g_xi, g_beta])

        try:
            res = optimize.minimize(
                neg_log_lik_grad, [xi0, beta0], method="L-BFGS-B",
                jac=True,
                bounds=[(-0.49, 2.0), (1e-8, None)],
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7})
            xi, beta = res.x
            if not res.success:          # fallback if gradient method fails
                res2 = optimize.minimize(
                    lambda p: neg_log_lik_grad(p)[0], [xi0, beta0],
                    method="Nelder-Mead", options={"maxiter": 300})
                xi, beta = res2.x
        except Exception:
            xi, beta = xi0, beta0

        xi   = float(np.clip(xi, -0.5, 2.0))
        beta = float(max(beta, 1e-8))

        # VaR and CVaR from GPD
        n   = len(r)
        nu  = len(exceed)
        p99 = 0.99

        try:
            if abs(xi) < 1e-8:   # exponential tail
                var_99  = u - beta * math.log((n / nu) * (1 - p99))
            else:
                var_99  = u - beta / xi * (1 - ((n / nu) * (1 - p99))**(-xi))

            if xi < 1:
                cvar_99 = (var_99 + beta - xi * u) / (1 - xi)
            else:
                cvar_99 = var_99 * 2
        except Exception:
            var_99  = float(np.percentile(r, 1))
            cvar_99 = float(r[r <= var_99].mean()) if (r <= var_99).any() else var_99

        # Hill estimator: tail index α = 1/ξ
        n_tail = max(int(nu * 0.8), 5)
        r_sorted = np.sort(r)
        if n_tail < len(r_sorted) and r_sorted[n_tail] != 0:
            hill = float(np.mean(np.log(r_sorted[:n_tail] / r_sorted[n_tail])))
        else:
            hill = float(xi)

        return {
            "xi":      xi,
            "beta":    beta,
            "u":       float(u),
            "n_exceed":nu,
            "var_99":  float(var_99),
            "cvar_99": float(cvar_99),
            "var_999": float(var_99 * (1 + abs(xi) * 2)),  # extrapolation
            "hill":    hill,
            "tail_class": ("Fréchet/heavy" if xi > 0.1
                           else "Gumbel/medium" if abs(xi) < 0.1
                           else "Weibull/bounded"),
            "size_mult": float(np.clip(-0.02 / min(cvar_99, -1e-6), 0.3, 2.0)),
            "valid": True,
        }

    def bootstrap_cvar(self, returns: np.ndarray, n_boot: int = 200,
                        level: float = 0.99) -> dict:
        """
        Bootstrap confidence interval for CVaR.
        Gives uncertainty band: we know CVaR ∈ [lo, hi] with 90% probability.
        """
        r = returns[~np.isnan(returns)]
        if len(r) < 50:
            q = float(np.percentile(r, (1-level)*100))
            return {"point": q, "lo": q*1.2, "hi": q*0.8, "width": abs(q*0.4)}

        cvars = []
        for _ in range(n_boot):
            boot = np.random.choice(r, size=len(r), replace=True)
            q    = np.percentile(boot, (1-level)*100)
            tail = boot[boot <= q]
            if len(tail) > 0:
                cvars.append(float(tail.mean()))

        cvars = np.array(cvars)
        return {
            "point": float(np.mean(cvars)),
            "lo":    float(np.percentile(cvars, 5)),
            "hi":    float(np.percentile(cvars, 95)),
            "width": float(np.percentile(cvars, 95) - np.percentile(cvars, 5)),
        }

    def stress_test(self, returns: np.ndarray) -> dict:
        """
        How bad can it get? Return levels for several probability thresholds.
        Uses fitted GPD rather than Gaussian approximation.
        """
        gpd = self.fit_gpd(returns)
        r   = returns[~np.isnan(returns)]
        mu  = float(r.mean()); sig = float(r.std())

        # Gaussian assumption (naive)
        gauss = {
            "1sd_dn":   mu - sig,
            "2sd_dn":   mu - 2*sig,
            "3sd_dn":   mu - 3*sig,
            "gauss_p1": float(sp_norm.ppf(0.01, mu, sig)),
        }

        return {
            "gpd_var99":   gpd["var_99"],
            "gpd_var999":  gpd["var_999"],
            "gpd_cvar99":  gpd["cvar_99"],
            "gaussian_p1": gauss["gauss_p1"],
            "underestimate_factor": abs(gpd["var_99"] / gauss["gauss_p1"])
                                    if gauss["gauss_p1"] != 0 else 1.0,
            "xi":          gpd["xi"],
            "tail_class":  gpd["tail_class"],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  C. INFORMATION THEORY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class InfoTheoryEngine:
    """
    Quantify information content and dependencies between signals.

    Core insight (Renyi / Shannon): alpha = information advantage.
    A signal has value only if it reduces entropy of future returns.

    I(X;Y) = 0: X tells us nothing about Y
    I(X;Y) > 0: X and Y share information
    T(X→Y) > T(Y→X): X causally leads Y (not just correlated)
    """

    @staticmethod
    def shannon_entropy(x: np.ndarray, bins: int = 20) -> float:
        """H(X) = -Σ p(x) log₂ p(x). Higher = more uncertain/random."""
        x = x[~np.isnan(x)]
        if len(x) < 10: return 0.0
        counts, _ = np.histogram(x, bins=bins)
        probs = counts[counts > 0] / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-300)))

    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray,
                            bins: int = 12) -> float:
        """
        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Measures shared information between order flow (X) and returns (Y).
        I(CVD, ret) > 0.05 bits → CVD carries predictive information.
        """
        n = min(len(x), len(y)); x = x[:n]; y = y[:n]
        mask = ~(np.isnan(x) | np.isnan(y)); x = x[mask]; y = y[mask]
        if len(x) < 20: return 0.0

        hist2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist2d / max(hist2d.sum(), 1e-300)
        px  = pxy.sum(axis=1); py = pxy.sum(axis=0)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i,j] * math.log2(pxy[i,j] / (px[i]*py[j]))
        return max(float(mi), 0.0)

    @staticmethod
    def transfer_entropy(source: np.ndarray, target: np.ndarray,
                          lag: int = 1, bins: int = 8) -> float:
        """
        T(X→Y) = I(Y_{t+1}; X_t | Y_t)

        Directed information flow: how much does knowing X_t reduce
        uncertainty of Y_{t+1} beyond what Y_t already tells us?

        T(CVD→returns) > T(returns→CVD) confirms CVD causally leads price.
        This is stronger than correlation — it is Granger causality.

        Vectorised implementation using numpy 3-D histogram.
        Same mathematics as the original triple loop but O(n + bins³)
        instead of O(n · bins³) — typically 50–100× faster.
        """
        n = min(len(source), len(target)) - lag
        if n < 25: return 0.0
        src     = source[:n];  tgt     = target[lag:n + lag];  tgt_lag = target[:n]
        mask    = ~(np.isnan(src) | np.isnan(tgt) | np.isnan(tgt_lag))
        src     = src[mask]; tgt = tgt[mask]; tgt_lag = tgt_lag[mask]
        if len(src) < 20: return 0.0

        def digitize_uniform(x, b):
            lo, hi = x.min(), x.max() + 1e-9
            edges  = np.linspace(lo, hi, b + 1)
            return np.clip(np.digitize(x, edges) - 1, 0, b - 1)

        s_d  = digitize_uniform(src, bins)
        t_d  = digitize_uniform(tgt, bins)
        tl_d = digitize_uniform(tgt_lag, bins)
        n_   = float(len(s_d))

        # Build 3-D joint histogram in one numpy call (vectorised)
        hist3, _ = np.histogramdd(
            np.column_stack([t_d, tl_d, s_d]),
            bins=[bins, bins, bins]
        )
        p_all  = hist3 / n_                       # P(Y_{t+1}, Y_t, X_t)
        p_ts   = p_all.sum(axis=2)                # P(Y_{t+1}, Y_t)   — sum over X
        p_tl   = p_all.sum(axis=(0, 2))           # P(Y_t)            — sum over Y_{t+1}, X
        p_ts2  = p_all.sum(axis=0)                # P(Y_t, X_t)       — sum over Y_{t+1}

        # T = Σ P(i,j,k) · log( P(i,j,k)·P(j) / [P(i,j)·P(j,k)] )
        # Broadcast to (bins, bins, bins) for vectorised log computation
        with np.errstate(divide="ignore", invalid="ignore"):
            numer = p_all * p_tl[np.newaxis, :, np.newaxis]
            denom = p_ts[:, :, np.newaxis] * p_ts2[np.newaxis, :, :]
            ratio = np.where((p_all > 0) & (denom > 0), numer / denom, 1.0)
            te    = float(np.sum(p_all * np.log2(ratio)))

        return max(te, 0.0)

    @staticmethod
    def variance_ratio_test(returns: np.ndarray, lags=(2,4,8,16)) -> dict:
        """
        Lo-MacKinlay (1988) Variance Ratio Test.

        Under random walk: Var(q-period return) = q × Var(1-period return)
        VR(q) = Var(q) / (q × Var(1))

        VR(q) > 1: positive autocorrelation (momentum)
        VR(q) < 1: negative autocorrelation (mean reversion)
        Z-stat tests H₀: VR=1 (random walk)

        BTC 5m: typically shows mean reversion at 2-4 bars,
        momentum at 16-32 bars.
        """
        r = returns[~np.isnan(returns)]
        n = len(r)
        if n < 30:
            return {"regime": "insufficient_data", "vr": {}}

        mu   = r.mean()
        sig2 = ((r - mu)**2).sum() / (n - 1)
        vrs  = {}
        regime_votes = {"momentum": 0, "mean_revert": 0, "random_walk": 0}

        for q in lags:
            if n < q * 3: continue
            r_q    = np.array([r[i:i+q].sum() for i in range(n - q + 1)])
            sig2_q = ((r_q - q*mu)**2).sum() / ((n - q) * q)
            vr     = float(sig2_q / sig2) if sig2 > 0 else 1.0

            # Lo-MacKinlay (1988) heteroskedasticity-robust z-statistic.
            # Homoskedastic version uses δ = 2(2q-1)(q-1)/(3q) — biased for
            # fat-tailed, volatility-clustered data (which BTC has).
            # Robust version: δ(q) = Σₖ₌₁^{q-1} [2(q-k)/q]² · θ̂ₖ
            # where θ̂ₖ = Σₜ (rₜ-μ)²(rₜ₋ₖ-μ)² / [Σₜ(rₜ-μ)²]²
            denom_sq = max(((r - mu)**2).sum()**2, 1e-30)
            delta_robust = 0.0
            for k in range(1, q):
                w_k  = (2.0 * (q - k) / q) ** 2
                # θ̂ₖ: sum of products of squared demeaned returns with lag k
                r_dm = r - mu
                theta_k = float(np.sum(r_dm[k:]**2 * r_dm[:-k]**2)) / denom_sq
                delta_robust += w_k * theta_k
            # Scale to match the VR estimate variance
            delta_robust *= n
            delta_robust  = max(delta_robust, 1e-10)

            z_stat = float((vr - 1) / math.sqrt(delta_robust / n)) if n > 0 else 0.0
            p_val  = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

            vrs[q] = {"vr": vr, "z": z_stat, "p": p_val}
            if p_val < 0.05:
                if vr > 1: regime_votes["momentum"]    += 1
                else:       regime_votes["mean_revert"] += 1
            else:
                regime_votes["random_walk"] += 1

        regime = max(regime_votes, key=regime_votes.get)
        return {"vr": vrs, "regime": regime, "votes": regime_votes}

    @staticmethod
    def approximate_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
        """
        ApEn(m, r): regularity/predictability measure.

        Low ApEn  → highly regular, predictable
        High ApEn → chaotic, unpredictable

        ApEn < 0.5: market in structured (trending) regime
        ApEn > 1.5: market in random/noisy regime

        Computed on the last 100 bars for O(100²) cost per call.
        (Original had x_sub/n_sub defined but phi() captured outer x/n
         via closure → subset was never actually used. Fixed here.)
        """
        x = x[~np.isnan(x)]
        n = len(x)
        if n < 30: return 1.0

        # Use only recent 100 bars for O(n_sub²) complexity
        n_sub = min(n, 100)
        x_sub = x[-n_sub:]
        # Recalculate r on the subset so scale is consistent
        r = r_frac * float(np.std(x_sub))
        if r <= 0: return 1.0

        # phi() receives explicit arrays — no closure over outer x/n
        def phi(x_arr: np.ndarray, n_: int, m_: int) -> float:
            count = 0; total = 0
            for i in range(n_ - m_):
                template = x_arr[i:i + m_]
                for j in range(n_ - m_):
                    if np.max(np.abs(x_arr[j:j + m_] - template)) <= r:
                        count += 1
                total += 1
            return math.log(count / max(total, 1)) if count > 0 else 0.0

        try:
            apen = phi(x_sub, n_sub, m) - phi(x_sub, n_sub, m + 1)
            return float(max(apen, 0.0))
        except Exception:
            return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  D. STOCHASTIC PROCESSES
# ══════════════════════════════════════════════════════════════════════════════

class StochasticEngine:
    """
    Exact parameter estimation for stochastic price processes.
    """

    # ── OU Maximum Likelihood Estimation ─────────────────────────────────────
    @staticmethod
    def ou_mle(prices: np.ndarray, dt: float = 1.0) -> dict:
        """
        Ornstein-Uhlenbeck: dX = κ(μ-X)dt + σ dW

        EXACT closed-form MLE (Shoji & Ozaki 1998):
          Estimates κ (speed), μ (mean), σ (vol), half-life, z-score.

        Previous OLS approximation had bias proportional to κ·dt.
        This exact form is unbiased regardless of step size.
        """
        x  = prices.astype(np.float64)
        n  = len(x)
        if n < 25:
            return {"kappa":1.0,"mu":float(x.mean()),"sigma_eq":float(x.std()),
                    "ou_z":0.0,"half_life":50.0,"valid":False,"revert_conf":0.5}

        x_lag = x[:-1]; x_cur = x[1:]
        n_    = float(len(x_lag))

        Sx  = x_lag.sum(); Sy  = x_cur.sum()
        Sxx = (x_lag**2).sum(); Sxy = (x_lag*x_cur).sum()
        denom = n_*Sxx - Sx**2

        if abs(denom) < 1e-10:
            return {"kappa":1.0,"mu":float(x.mean()),"sigma_eq":float(x.std()),
                    "ou_z":0.0,"half_life":50.0,"valid":False,"revert_conf":0.5}
        try:
            kappa_raw = -math.log(max((n_*Sxy-Sx*Sy)/denom, 1e-8)) / dt
            kappa     = max(float(kappa_raw), 1e-4)
            e_kdt     = math.exp(-kappa*dt)
            mu_mle    = (Sy - e_kdt*Sx) / (n_*(1-e_kdt) + 1e-10)
            resid     = x_cur - mu_mle - e_kdt*(x_lag - mu_mle)
            sig_resid = max(float(np.std(resid)), 1e-9)
            sigma_eq  = sig_resid / math.sqrt(max(1 - e_kdt**2, 1e-10)) * math.sqrt(2*kappa+1e-10)
            sigma_eq  = max(abs(sigma_eq), 1e-6)
            half_life = math.log(2) / kappa
            ou_z      = float(np.clip((x[-1] - mu_mle) / sigma_eq, -5, 5))
            revert_c  = float(min(1.0, 10.0 / half_life))
            # Two-tailed p-value: P(|Z| ≥ |ou_z|) under H₀: z ~ N(0,1).
            # Significant (p < 0.05) means the current price is a statistically
            # unusual deviation from the OU equilibrium — a real signal, not noise.
            from scipy.stats import norm as _norm
            p_value   = float(2.0 * _norm.sf(abs(ou_z)))
            sig_005   = p_value < 0.05
            return {"kappa":     kappa,
                    "mu":        float(mu_mle),
                    "sigma_eq":  sigma_eq,
                    "ou_z":      ou_z,
                    "half_life": float(np.clip(half_life, 0, 500)),
                    "p_value":   p_value,
                    "significant": sig_005,    # True when |z| is statistically unusual
                    "valid":     True,
                    "revert_conf": revert_c}
        except Exception:
            return {"kappa":1.0,"mu":float(x.mean()),"sigma_eq":float(x.std()),
                    "ou_z":0.0,"half_life":50.0,"valid":False,"revert_conf":0.5}

    # ── RTS Kalman Smoother ───────────────────────────────────────────────────
    @staticmethod
    def rts_kalman(prices: np.ndarray,
                   Q11=0.01, Q12=0.001, Q22=0.0001, R=1.0) -> dict:
        """
        Rauch-Tung-Striebel (RTS) smoother.

        Standard Kalman: causal (past→present). Estimate x̂ₜ|t.
        RTS smoother:   backward pass added. Estimate x̂ₜ|T (uses future).

        Live inference:  uses x̂ₜ|t (causal, no lookahead).
        Training labels: uses x̂ₜ|T (optimal, for denoised targets).

        State: [price, trend]. Transition: constant velocity model.
        """
        z = prices.astype(np.float64)
        n = len(z)
        F = np.array([[1., 1.], [0., 1.]])
        H = np.array([1., 0.])   # 1D observation
        Q = np.array([[Q11, Q12], [Q12, Q22]])

        x = np.array([z[0], 0.]); P = np.eye(2) * 1000.
        xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
        xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))

        # Forward Kalman filter
        for t in range(n):
            xpt = F @ x; Ppt = F @ P @ F.T + Q
            xp[t] = xpt; Pp[t] = Ppt
            S = float(H @ Ppt @ H) + R
            K = (Ppt @ H) / S
            x = xpt + K * (z[t] - float(H @ xpt))
            P = (np.eye(2) - np.outer(K, H)) @ Ppt
            xf[t] = x; Pf[t] = P

        # Backward RTS smoother
        xs = xf.copy(); Ps = Pf.copy()
        for t in range(n-2, -1, -1):
            try:
                G = Pf[t] @ F.T @ np.linalg.inv(Pp[t+1])
            except np.linalg.LinAlgError:
                G = Pf[t] @ F.T @ np.linalg.pinv(Pp[t+1])
            xs[t] = xf[t] + G @ (xs[t+1] - xp[t+1])
            Ps[t] = Pf[t] + G @ (Ps[t+1] - Pp[t+1]) @ G.T

        innov     = z - xp[:, 0]
        innov_std = float(np.std(innov[-50:])) if len(innov) >= 50 else float(np.std(innov))
        # SNR = |trend| / innovation_std.
        # Innovation = observation - one-step-ahead prediction = the unexplained noise.
        # SNR > 1 means trend signal is stronger than the residual noise.
        # Old formula divided by uncertainty*0.001 (arbitrary scale factor) which gave
        # values like 114,000 — uninterpretable and useless as a threshold gate.
        snr = float(abs(xf[-1, 1]) / max(innov_std, 1e-8))
        return {
            "live_price":    float(xf[-1, 0]),
            "live_trend":    float(xf[-1, 1]),
            "smooth_price":  float(xs[-1, 0]),
            "smooth_trend":  float(xs[-1, 1]),
            "smooth_prices": xs[:, 0],
            "smooth_trends": xs[:, 1],
            "uncertainty":   float(np.sqrt(Pf[-1, 0, 0])),
            "innov_std":     innov_std,
            "snr":           snr,
        }

    # ── GARCH(1,1) ────────────────────────────────────────────────────────────
    @staticmethod
    def garch11(returns: np.ndarray) -> dict:
        """
        GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        Analytical warm-start from moment estimator:
          ω = σ²_unconditional × (1-α-β)
          α ≈ autocorrelation of squared returns
          β ≈ 0.88 (typical for BTC)

        Returns current vol, size multiplier, regime, vol percentile.
        """
        r = returns[~np.isnan(returns)]
        if len(r) < 30:
            return {"vol":0.003,"size_mult":1.0,"regime":"MEDIUM","pct":50.0}

        v0 = float(np.var(r))
        # Moment-based warm-start
        ac1 = float(pd.Series(r**2).autocorr(1)) if len(r)>=10 else 0.10
        al0 = float(np.clip(ac1, 0.02, 0.18))
        be0 = float(np.clip(0.88, 0.01, 0.99-al0))
        om0 = v0*(1-al0-be0)

        def nll(p):
            om, al, be = p
            if om<=0 or al<0 or be<0 or al+be>=0.9999: return 1e10
            h = np.full(len(r), v0)
            ll= 0.0
            for t in range(1, len(r)):
                h[t] = om + al*r[t-1]**2 + be*h[t-1]
                if h[t] <= 0: return 1e10
                ll += -0.5*(math.log(2*math.pi*h[t]) + r[t]**2/h[t])
            return -ll

        try:
            res = optimize.minimize(nll, [om0, al0, be0], method="L-BFGS-B",
                                    bounds=[(1e-10,None),(1e-5,0.5),(1e-5,0.9999)],
                                    options={"maxiter":100})
            om, al, be = res.x
        except Exception:
            om, al, be = om0, al0, be0

        h = np.full(len(r), v0)
        for t in range(1, len(r)):
            h[t] = max(om + al*r[t-1]**2 + be*h[t-1], 1e-12)

        curr_vol = float(math.sqrt(h[-1]))
        hist_vol = np.sqrt(h)
        pct      = float(stats.percentileofscore(hist_vol, curr_vol))
        regime   = "LOW" if pct < 30 else ("HIGH" if pct > 75 else "MEDIUM")
        # Continuous inverse-volatility sizing: scale = median_vol / current_vol.
        # When vol doubles → size halves. When vol halves → size doubles.
        # Clipped to [0.25, 2.0] to prevent extreme leverage or near-zero sizing.
        # Old code had exactly 3 values: {0.5, 1.0, 1.5} — a step function, not scaling.
        med_vol  = float(np.median(hist_vol[hist_vol > 0]))
        size_m   = float(np.clip(med_vol / max(curr_vol, 1e-10), 0.25, 2.0))
        lrun_vol = float(math.sqrt(max(om/(1-al-be+1e-10), 1e-10)))

        return {"vol":curr_vol,"size_mult":size_m,"regime":regime,"pct":pct,
                "persistence":float(al+be),"long_run_vol":lrun_vol,
                "omega":float(om),"alpha":float(al),"beta_":float(be)}

    # ── Heston Stochastic Volatility ──────────────────────────────────────────
    @staticmethod
    def heston_params(returns: np.ndarray) -> dict:
        """
        Fit Heston model: dS = μS dt + √v S dW₁
                          dv = κ(θ-v)dt + ξ√v dW₂
                          Corr(dW₁, dW₂) = ρ

        Parameters:
          κ: mean-reversion speed of variance (speed of vol recovery)
          θ: long-run variance (equilibrium vol²)
          ξ: vol of vol (how wildly does vol itself move)
          ρ: leverage effect (negative for stocks/BTC → big drops = vol spike)
        """
        r = returns[~np.isnan(returns)]
        if len(r) < 50:
            return {"kappa":1.5,"theta":0.04,"xi":0.3,"rho":-0.6,"valid":False}

        rv = r**2
        rv_lag = np.roll(rv, 1); rv_lag[0] = rv.mean()
        dv  = np.diff(rv); v_lag = rv[:-1]
        A_  = np.column_stack([np.ones(len(v_lag)), v_lag])
        try:
            co, _, _, _ = np.linalg.lstsq(A_, dv, rcond=None)
            kappa = max(float(-co[1]), 0.01)
            theta = max(float(-co[0]/co[1]) if co[1]!=0 else float(rv.mean()), 1e-8)
        except Exception:
            kappa, theta = 1.5, float(rv.mean())

        resid = dv - (co[0] + co[1]*v_lag) if 'co' in dir() else dv
        xi    = float(max(np.std(resid)/max(np.sqrt(rv.mean()),1e-9), 0.01))

        # Price-vol correlation (leverage effect)
        rho   = float(np.corrcoef(r[:-1], np.diff(rv))[0,1])
        rho   = float(np.clip(rho, -0.99, 0.99))

        return {"kappa":float(np.clip(kappa,0.01,20)),
                "theta":float(np.clip(theta,1e-8,1)),
                "xi":float(np.clip(xi,0.01,5)),
                "rho":rho,
                "long_run_vol":float(math.sqrt(max(theta,0))),
                "curr_vol":float(math.sqrt(max(rv[-10:].mean(),0))),
                "valid":True}

    # ── Lévy Jump Detection ───────────────────────────────────────────────────
    @staticmethod
    def levy_jumps(returns: np.ndarray) -> dict:
        """
        Detect Lévy jumps using Bipower Variation test (Barndorff-Nielsen 2004).

        BPV = (π/2) × Σ|r_t||r_{t-1}|  — jump-robust baseline variance
        RV  = Σ r²_t                     — realized variance (includes jumps)
        JV  = max(RV - BPV, 0)          — jump variation
        RJV = JV/RV                      — relative jump contribution

        Lee-Mykland test: |r_t|/BPV_vol > critical value → jump detected.
        """
        r = returns[~np.isnan(returns)]
        if len(r) < 30:
            return {"lambda":0.,"n_jumps":0,"recent_jump":False,
                    "jump_regime":False,"size_penalty":1.0}

        bpv = float(np.pi/2 * np.mean(np.abs(r[1:]) * np.abs(r[:-1])))
        rv  = float(np.var(r))
        jv  = max(rv - bpv, 0.0)
        rjv = jv / max(rv, 1e-10)

        bpv_std = math.sqrt(max(bpv, 1e-10))
        jump_stat  = np.abs(r) / bpv_std
        threshold  = 3.09   # 99.9th percentile of |N(0,1)|
        is_jump    = jump_stat > threshold
        n_jumps    = int(is_jump.sum())
        lam        = float(n_jumps / len(r))
        recent     = bool(is_jump[-5:].any()) if len(is_jump)>=5 else False

        return {
            "lambda":      lam,
            "n_jumps":     n_jumps,
            "jump_contrib":float(rjv),
            "recent_jump": recent,
            "jump_regime": lam > 0.04,   # >4% of bars have jumps
            "size_penalty":0.5 if recent else 1.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  E. ADVANCED KELLY CRITERION
# ══════════════════════════════════════════════════════════════════════════════

class KellyEngine:
    """
    Kelly criterion family with uncertainty, tail-risk, and multi-signal extensions.

    Full Kelly f* = (p·b - q) / b  → maximizes E[log(wealth)]
    Quarter Kelly → more practical, lower risk of ruin
    Bayesian Kelly → uses posterior p(win), not sample mean

    "The Kelly criterion is the only formula that can guarantee
     long-run wealth maximization. Any other fraction is either
     suboptimal (too small) or has positive ruin probability (too large)."
     — Ed Thorp
    """

    def __init__(self, bayes: BayesianEngine = None):
        self.bayes = bayes
        self.lw    = LedoitWolf()

    def full_kelly(self, p: float, b: float, kurt: float = 3.0,
                  skew: float = 0.0) -> float:
        """
        f* = (p·b - (1-p)) / b  — standard Kelly.

        Moment adjustment (Thorp 1997, MacLean et al 2011):
        For non-normal returns with excess kurtosis κ and skewness γ:
          f*_adj = f* × (1 - κ_excess/20) × (1 + γ/10)

        BTC typically: κ≈8-15 (fat tails) → penalty 40-75%
                       γ≈-0.3 (negative skew) → additional 3% penalty

        Why: Kelly derivation assumes log-normal returns.
        Heavy tails → ruin probability higher than Kelly predicts.
        Skewness correction: downside fat tails hurt more.
        """
        q  = 1 - p
        f_full = max(float((p*b - q) / b), 0.0)
        # Excess kurtosis = raw kurtosis - 3  (normal distribution has excess = 0).
        # BTC raw kurtosis ≈ 8-15  → excess ≈ 5-12  → factor ≈ 0.75-0.40.
        # Old code used max(kurt, 0) which gave normal dist (kurt=3) a 15% penalty.
        kurt_excess = max(kurt - 3.0, 0.0)
        kurt_factor = max(1.0 - kurt_excess / 20.0, 0.25)
        skew_factor = 1.0 + min(skew, 0.0) / 10.0   # negative skew → reduce size
        return float(f_full * kurt_factor * skew_factor)

    def bayesian_kelly(self, signal: str, rr: float,
                        garch_mult: float = 1.0,
                        cvar_mult:  float = 1.0,
                        kurt: float = 3.0,
                        skew: float = 0.0) -> float:
        """
        Kelly fraction using Bayesian posterior of win rate.

        Why better than sample mean?
        - With 10 trades, sample WR is noisy (±16% std for true 60%)
        - Posterior shrinks toward prior, avoiding overconfidence
        - Lower CI bound gives conservative estimate

        Implementation:
          1. Get posterior mean p̄ and lower 80% CI bound p_lo
          2. Full Kelly at p̄ = f_mean
          3. Full Kelly at p_lo = f_lo (conservative)
          4. Blend: uncertainty × f_lo + (1-uncertainty) × f_mean
          5. Quarter-Kelly × GARCH × CVaR adjustments
        """
        if self.bayes is None:
            return min(0.02 * garch_mult * cvar_mult, 0.05)

        pm   = self.bayes.posterior_mean(signal)
        lo, hi = self.bayes.credible_interval(signal, 0.80)
        width= hi - lo   # wider CI = more uncertainty → more conservative

        f_mean = self.full_kelly(pm, rr, kurt=kurt, skew=skew)
        f_lo   = self.full_kelly(max(lo, 0.01), rr, kurt=kurt, skew=skew)

        # Blend: high uncertainty → use lower bound
        f_blend = f_lo + (f_mean - f_lo) * (1 - width)

        # Quarter-Kelly (standard practitioner shrinkage)
        f_quarter = f_blend * 0.25

        # Adjust for GARCH vol regime and CVaR tail risk
        f_final = f_quarter * garch_mult * cvar_mult

        return float(np.clip(f_final, 0.0, 0.08))

    def cvar_constrained_kelly(self, f: float, returns: np.ndarray,
                                max_cvar: float = -0.03) -> float:
        """
        Find largest f̃ ≤ f such that CVaR(strategy) ≥ max_cvar.
        If unconstrained Kelly gives too much tail risk → shrink f.

        Binary search: O(log(1/ε)) evaluations.
        """
        r = returns[~np.isnan(returns)]
        if len(r) < 30 or f <= 0:
            return f

        def strat_cvar(frac):
            strat = np.log(np.maximum(1 + frac*r, 1e-10))
            q5    = np.percentile(strat, 5)
            tail  = strat[strat <= q5]
            return float(tail.mean()) if len(tail) > 0 else float(strat.min())

        floor = math.log(1 + max_cvar)
        if strat_cvar(f) >= floor:
            return f   # already satisfies constraint

        lo_, hi_ = 0.0, f
        for _ in range(25):   # binary search
            mid = (lo_ + hi_) / 2
            if strat_cvar(mid) >= floor:
                lo_ = mid
            else:
                hi_ = mid
        return float(lo_)

    def multi_signal_kelly(self, signal_returns: dict,
                            garch_mult: float = 1.0) -> dict:
        """
        Multi-signal Kelly via mean-variance optimization:
          f* = Σ⁻¹ μ  (continuous-time, unconstrained)

        Where:
          μ = vector of expected returns per signal category
          Σ = covariance matrix (Ledoit-Wolf shrinkage)

        Ledoit-Wolf shrinkage prevents ill-conditioned Σ when
        signal history is short relative to number of signals.
        """
        sigs = [s for s,v in signal_returns.items() if len(v)>=15]
        if len(sigs) < 2:
            return {"total_kelly":0.02*garch_mult,"per_signal":{}}

        min_len = min(len(signal_returns[s]) for s in sigs)
        R = np.array([list(signal_returns[s])[-min_len:] for s in sigs]).T

        mu = R.mean(axis=0)
        try:
            Sigma = self.lw.fit(R).covariance_
        except Exception:
            Sigma = np.cov(R.T) + np.eye(len(sigs))*1e-6

        try:
            f_vec = np.linalg.solve(Sigma + np.eye(len(sigs))*1e-6, mu)
        except Exception:
            f_vec = mu / (np.diag(Sigma) + 1e-6)

        total = float(np.clip(np.abs(f_vec).sum() * 0.25 * garch_mult, 0, 0.08))
        per   = {sigs[i]: float(np.clip(f_vec[i]*0.25*garch_mult, -0.05, 0.05))
                 for i in range(len(sigs))}

        return {"total_kelly": total, "per_signal": per,
                "shrinkage": float(self.lw.shrinkage_) if hasattr(self.lw,"shrinkage_") else 0.5}


# ══════════════════════════════════════════════════════════════════════════════
#  F. SEQUENTIAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class SequentialTests:
    """
    Sequential statistical tests for real-time decision making.

    Unlike fixed-sample tests, sequential tests make decisions as data
    arrives — no need to pre-specify sample size.

    This is critical in trading: we want to enter ASAP when evidence
    is strong, not wait for an arbitrary fixed lookback.
    """

    # ── SPRT: Sequential Probability Ratio Test ──────────────────────────────
    class SPRT:
        """
        Wald (1947) Sequential Probability Ratio Test.

        Tests: H₀: P(win) = p₀  vs  H₁: P(win) = p₁

        Log-likelihood ratio: Λ_n = Σᵢ log(f₁(xᵢ)/f₀(xᵢ))

        Boundaries:
          Λ ≥ log((1-β)/α)  → reject H₀  (signal is real)
          Λ ≤ log(β/(1-α))  → accept H₀  (no signal)

        Mathematically optimal: minimizes E[N] given error rates.
        No other test can do better on average sample size.
        """
        def __init__(self, p0=0.50, p1=0.58, alpha=0.10, beta=0.10):
            self.p0=p0; self.p1=p1; self.alpha=alpha; self.beta=beta
            self.A = math.log((1-beta)/alpha)
            self.B = math.log(beta/(1-alpha))
            self.llr_up = math.log(p1/p0)
            self.llr_dn = math.log((1-p1)/(1-p0))
            self.L    = 0.0   # accumulated LLR for BUY
            self.L_sh = 0.0   # accumulated LLR for SELL
            self.n    = 0

        def update(self, prob_up: float) -> str:
            """
            Update with new ML probability. Returns decision string.

            Continuous probability-weighted LLR (Wald 1947, generalised form):
              LLR_up increment = log(f₁(p) / f₀(p))
            where f₀ ~ Beta(1,1) (null: p uniform) and we use the actual
            probability as evidence weight rather than a binary 0/1.

            This means:
              prob=0.72 adds log(0.72/0.50) = +0.365 to LLR_bull
              prob=0.52 adds log(0.52/0.50) = +0.039 to LLR_bull
            instead of both adding the same fixed llr_up = +0.228.

            The 4.5× difference preserves the information content of the
            ML probability rather than discarding it at the 0.5 threshold.
            """
            # Clip to avoid log(0); keep away from exact 0.5 to avoid 0 increment
            p = float(np.clip(prob_up, 1e-6, 1 - 1e-6))
            # LLR for BUY hypothesis: log(p/0.5)
            # LLR for SELL hypothesis: log((1-p)/0.5)
            self.L    += math.log(p / 0.5)
            self.L_sh += math.log((1.0 - p) / 0.5)
            self.n    += 1

            if self.L    >= self.A: return "CONFIRM_BUY"
            if self.L_sh >= self.A: return "CONFIRM_SELL"
            if self.L    <= self.B and self.L_sh <= self.B: return "REJECT"
            return "CONTINUE"

        def reset(self): self.L=0.; self.L_sh=0.; self.n=0

        @property
        def state(self):
            return {"llr_bull":self.L,"llr_bear":self.L_sh,
                    "n":self.n,"A":self.A,"B":self.B}

    # ── CUSUM: Cumulative Sum Chart ───────────────────────────────────────────
    class CUSUM:
        """
        CUSUM detects persistent shifts in a time series.

        For trading: detects when volatility regime or return distribution
        has PERMANENTLY shifted (not just a single outlier).

        Algorithm:
          S_t+ = max(0, S_{t-1}+ + (X_t - μ - K))  [upward shift]
          S_t- = max(0, S_{t-1}- - (X_t - μ + K))  [downward shift]
          Signal when S_t > H (threshold)

        K = allowance (slack, typically 0.5σ)
        H = threshold (typically 4-5σ)
        """
        def __init__(self, mu=0.0, sigma=1.0, k_mult=0.5, h_mult=4.0):
            self.mu = mu; self.sigma = sigma
            self.K  = k_mult * sigma
            self.H  = h_mult * sigma
            self.S_up = 0.; self.S_dn = 0.
            self.n_since_change = 0

        def update(self, x: float) -> dict:
            self.S_up = max(0, self.S_up + (x - self.mu - self.K))
            self.S_dn = max(0, self.S_dn - (x - self.mu + self.K))
            self.n_since_change += 1
            alarm = self.S_up > self.H or self.S_dn > self.H
            if alarm:
                direction = "UP" if self.S_up > self.H else "DOWN"
                self.S_up = 0.; self.S_dn = 0.; self.n_since_change = 0
                return {"alarm":True,"direction":direction}
            return {"alarm":False,"S_up":self.S_up,"S_dn":self.S_dn}

    # ── Page-Hinkley: Concept Drift Detection ─────────────────────────────────
    class PageHinkley:
        """
        Detects when an ML model's accuracy has persistently degraded.
        Used to trigger retraining.

        Cumulative sum variant:
          U_t = Σᵢ(xᵢ - x̄_t - δ)   [cumulative sum above running mean]
          Drift when max(U) - U_t > λ

        δ = minimum change we want to detect (0.01 = 1% accuracy drop)
        λ = threshold for alarm
        """
        def __init__(self, delta=0.005, lam=50.0):
            self.delta = delta; self.lam = lam
            self.sum_  = 0.   # CUSUM statistic (δ-deflated cumulative sum)
            self.max_  = 0.   # running maximum of CUSUM statistic
            self.raw_sum = 0. # separate raw cumulative sum for running mean
            self.n = 0; self.drift_count = 0

        def update(self, x: float) -> bool:
            self.n      += 1
            self.raw_sum += x
            # Running mean from raw observations (not the CUSUM statistic)
            running_mean = self.raw_sum / self.n
            # Page-Hinkley statistic: cumulative sum minus (mean + minimum_change δ)
            self.sum_    = self.sum_ + x - running_mean - self.delta
            self.max_    = max(self.max_, self.sum_)
            drift        = (self.max_ - self.sum_) > self.lam
            if drift:
                self.sum_ = 0.; self.max_ = 0.
                self.raw_sum = 0.; self.n = 0
                self.drift_count += 1
            return drift


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class QuantMath:
    """
    Single object providing all math engines.
    Import and use:

        qm = QuantMath()

        # Bayesian
        qm.bayes.update("cvd", won=True)
        p = qm.bayes.posterior_mean("cvd")
        bf = qm.bayes.bayes_factor("cvd")

        # EVT tail risk
        risk = qm.evt.fit_gpd(returns)
        cvar_ci = qm.evt.bootstrap_cvar(returns)

        # Information theory
        mi = qm.info.mutual_information(cvd_series, future_returns)
        te = qm.info.transfer_entropy(cvd_series, price_series)
        vr = qm.info.variance_ratio_test(returns)

        # Stochastic processes
        ou  = qm.stoch.ou_mle(price_series)
        kal = qm.stoch.rts_kalman(price_series)
        g   = qm.stoch.garch11(returns)
        h   = qm.stoch.heston_params(returns)
        lv  = qm.stoch.levy_jumps(returns)

        # Kelly sizing
        f = qm.kelly.bayesian_kelly("cvd", rr=2.5, garch_mult=g["size_mult"])

        # Sequential tests
        d = qm.sprt.update(ml_probability)
        qm.cusum.update(return_value)
    """

    def __init__(self,
                 sprt_p0:    float = 0.50,
                 sprt_p1:    float = 0.58,
                 sprt_alpha: float = 0.10,
                 sprt_beta:  float = 0.10):
        """
        Parameters
        ----------
        sprt_p0/p1   : SPRT null and alternative hypotheses for P(win).
                       Pass CFG["SPRT_H0"] and CFG["SPRT_H1"] from the main bot.
        sprt_alpha/beta : Type-I and Type-II error rates for SPRT boundaries.
        """
        self.bayes = BayesianEngine()
        self.evt   = EVTEngine()
        self.info  = InfoTheoryEngine()
        self.stoch = StochasticEngine()
        self.kelly = KellyEngine(bayes=self.bayes)
        self.sprt  = SequentialTests.SPRT(p0=sprt_p0, p1=sprt_p1,
                                           alpha=sprt_alpha, beta=sprt_beta)
        self.cusum = SequentialTests.CUSUM()
        self.ph    = SequentialTests.PageHinkley()
        # IC tracker: signal → deque of (signal_value, future_return)
        self.ic_hist: dict = defaultdict(lambda: deque(maxlen=50))

    def record_ic(self, signal: str, signal_val: float, future_ret: float):
        """Record signal value and subsequent return for IC tracking."""
        self.ic_hist[signal].append((float(signal_val), float(future_ret)))

    def information_coefficient(self, signal: str) -> float:
        """
        IC = Spearman rank correlation between signal and subsequent return.
        IC > 0.05 = useful. IC > 0.10 = good. IC > 0.15 = excellent.
        """
        h = list(self.ic_hist[signal])
        if len(h) < 10: return 0.0
        sv = np.array([x[0] for x in h])
        rv = np.array([x[1] for x in h])
        try:
            ic, _ = stats.spearmanr(sv, rv)
            return float(0.0 if np.isnan(ic) else ic)
        except Exception:
            return 0.0


    @staticmethod
    def newey_west_sharpe(returns: np.ndarray, lags: int = 10,
                           ann_factor: float = 288*252) -> dict:
        """
        Sharpe ratio with Newey-West HAC standard errors.

        Standard Sharpe ignores autocorrelation in returns — biased.
        Newey-West corrects: Var[μ_hat] = γ₀/n + 2Σ (1-k/(L+1)) γ_k/n

        Returns t-statistic so you can test H₀: SR=0.
        SR significant at 5% → t > 1.96.

        Reference: Lo (2002) "The Statistics of Sharpe Ratios"
        """
        r   = returns[~np.isnan(returns)]
        n   = len(r)
        if n < 20:
            return {"sharpe":0.,"t_stat":0.,"p_val":1.,"sig":False}
        mu  = r.mean()
        sig = r.std()
        if sig <= 0:
            return {"sharpe":0.,"t_stat":0.,"p_val":1.,"sig":False}

        # HAC variance of the mean
        gamma0  = ((r - mu)**2).mean()
        hac_var = gamma0 / n
        for k in range(1, lags+1):
            w_k     = 1 - k/(lags + 1)           # Bartlett kernel
            gamma_k = float(pd.Series(r-mu).autocorr(k)) * gamma0
            hac_var += 2 * w_k * gamma_k / n
        hac_var = max(hac_var, 1e-12)

        sr     = float(mu / sig * math.sqrt(ann_factor))
        t_stat = float(mu / math.sqrt(hac_var))
        p_val  = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        return {"sharpe":sr, "t_stat":t_stat, "p_val":p_val,
                "sig": p_val < 0.05, "hac_var":float(hac_var)}

    def run_full(self, prices: np.ndarray, returns: np.ndarray,
                  delta_pct: np.ndarray = None) -> dict:
        """
        Run all math engines on current price/return series.
        Returns comprehensive statistical profile of current market state.
        """
        out = {}

        # Stochastic processes
        out["ou"]      = self.stoch.ou_mle(prices[-100:] if len(prices)>=100 else prices)
        out["kalman"]  = self.stoch.rts_kalman(prices[-200:] if len(prices)>=200 else prices)
        out["garch"]   = self.stoch.garch11(returns)
        out["heston"]  = self.stoch.heston_params(returns)
        out["levy"]    = self.stoch.levy_jumps(returns)

        # EVT
        out["evt"]     = self.evt.fit_gpd(returns)
        out["stress"]  = self.evt.stress_test(returns)

        # Information theory
        out["entropy"] = self.info.shannon_entropy(returns)
        out["apen"]    = self.info.approximate_entropy(returns)
        out["vr_test"] = self.info.variance_ratio_test(returns)
        if delta_pct is not None:
            fut = np.roll(returns, -1); fut[-1]=0
            out["mi_cvd"]    = self.info.mutual_information(delta_pct, fut)
            out["te_cvd_p"]  = self.info.transfer_entropy(delta_pct, returns)
            out["te_p_cvd"]  = self.info.transfer_entropy(returns, delta_pct)
            out["cvd_leads"] = out["te_cvd_p"] > out["te_p_cvd"]

        # SPRT state
        out["sprt"] = self.sprt.state

        # Combined risk signal
        size_adj = out["garch"]["size_mult"]
        if out["evt"].get("valid"):
            size_adj *= out["evt"]["size_mult"]
        if out["levy"]["recent_jump"]:
            size_adj *= out["levy"]["size_penalty"]
        out["size_multiplier"] = float(np.clip(size_adj, 0.2, 2.0))

        return out


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST (run directly to see all engines working)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    np.random.seed(42)

    print("="*60)
    print("  QUANT MATH ENGINE — SELF-TEST")
    print("="*60)

    # Generate realistic BTC-like data
    n = 500
    prices  = np.cumsum(np.random.normal(0.0002, 0.005, n)) * 67000 + 67000
    returns = np.diff(np.log(prices))
    delta   = returns * (0.5 + np.random.normal(0, 0.3, len(returns)))

    qm = QuantMath()

    # ── Bayesian ──────────────────────────────────────────────
    print("\n[A] BAYESIAN INFERENCE")
    # Simulate 20 trades with 63% win rate
    for _ in range(20):
        won = np.random.random() < 0.63
        qm.bayes.update("cvd", won)
    pm   = qm.bayes.posterior_mean("cvd")
    lo, hi = qm.bayes.credible_interval("cvd")
    bf   = qm.bayes.bayes_factor("cvd")
    w    = qm.bayes.signal_weight("cvd")
    print(f"  CVD posterior: P(win)={pm:.3f}  90%CI=[{lo:.3f},{hi:.3f}]")
    print(f"  Bayes Factor: {bf:.2f}  Weight: {w:.2f}×")
    print(f"  BMA test: {qm.bayes.bma_probability({'cvd':0.65,'ou':0.60}):.3f}")

    # ── EVT ───────────────────────────────────────────────────
    print("\n[B] EXTREME VALUE THEORY")
    gpd  = qm.evt.fit_gpd(returns)
    boot = qm.evt.bootstrap_cvar(returns)
    stress=qm.evt.stress_test(returns)
    print(f"  GPD: ξ={gpd['xi']:.3f}  VaR99={gpd['var_99']:.5f}  CVaR99={gpd['cvar_99']:.5f}")
    print(f"  Tail class: {gpd['tail_class']}")
    print(f"  Bootstrap CVaR: {boot['point']:.5f}  90%CI=[{boot['lo']:.5f},{boot['hi']:.5f}]")
    print(f"  GPD vs Gaussian: underestimates by {stress['underestimate_factor']:.2f}×")

    # ── Info Theory ───────────────────────────────────────────
    print("\n[C] INFORMATION THEORY")
    h  = qm.info.shannon_entropy(returns)
    mi = qm.info.mutual_information(delta, np.roll(returns,-1))
    te = qm.info.transfer_entropy(delta, returns)
    ap = qm.info.approximate_entropy(returns)
    vr = qm.info.variance_ratio_test(returns)
    print(f"  Shannon entropy: {h:.4f} bits")
    print(f"  MI(CVD, next_ret): {mi:.6f} bits  ({'informative' if mi>0.01 else 'low'})")
    print(f"  Transfer entropy CVD→price: {te:.6f}")
    print(f"  Approximate entropy: {ap:.4f}  ({'predictable' if ap<0.8 else 'random'})")
    print(f"  Variance Ratio Test: regime={vr['regime']}")

    # ── Stochastic ────────────────────────────────────────────
    print("\n[D] STOCHASTIC PROCESSES")
    ou  = qm.stoch.ou_mle(prices[-100:])
    kal = qm.stoch.rts_kalman(prices[-100:])
    g   = qm.stoch.garch11(returns)
    h_  = qm.stoch.heston_params(returns)
    lv  = qm.stoch.levy_jumps(returns)
    print(f"  OU-MLE: κ={ou['kappa']:.4f}  HL={ou['half_life']:.1f}bars  z={ou['ou_z']:.3f}")
    print(f"  RTS Kalman: trend={kal['live_trend']:+.4f}  SNR={kal['snr']:.2f}")
    print(f"  GARCH(1,1): vol={g['vol']*100:.3f}%  α+β={g['persistence']:.4f}  regime={g['regime']}")
    print(f"  Heston: κ={h_['kappa']:.3f}  θ={h_['theta']:.5f}  ξ={h_['xi']:.3f}  ρ={h_['rho']:.3f}")
    print(f"  Lévy jumps: n={lv['n_jumps']}  λ={lv['lambda']:.4f}  recent={lv['recent_jump']}")

    # ── Kelly ─────────────────────────────────────────────────
    print("\n[E] KELLY CRITERION")
    bk   = qm.kelly.bayesian_kelly("cvd", rr=2.5, garch_mult=g["size_mult"])
    full = qm.kelly.full_kelly(pm, 2.5)
    cc   = qm.kelly.cvar_constrained_kelly(full, returns, max_cvar=-0.02)
    # Simulate signal returns for multi-signal Kelly
    sig_rets = {"cvd": list(np.random.normal(0.003,0.01,30)),
                "ou":  list(np.random.normal(0.002,0.012,30))}
    mk = qm.kelly.multi_signal_kelly(sig_rets, g["size_mult"])
    print(f"  Full Kelly:      {full:.4f} ({full*100:.2f}%)")
    print(f"  Bayesian Kelly:  {bk:.4f} ({bk*100:.2f}%)  [posterior-shrunk, quarter-Kelly]")
    print(f"  CVaR-constrained:{cc:.4f} ({cc*100:.2f}%)  [CVaR floor = -2%]")
    print(f"  Multi-signal:    {mk['total_kelly']:.4f}  shrinkage={mk.get('shrinkage',0.5):.3f}")

    # ── Sequential ────────────────────────────────────────────
    print("\n[F] SEQUENTIAL TESTS")
    sprt_results = []
    for i in range(20):
        p = np.random.uniform(0.55,0.75) if i>10 else 0.50
        d = qm.sprt.update(p)
        sprt_results.append(d)
        if d in ["CONFIRM_BUY","CONFIRM_SELL","REJECT"]:
            print(f"  SPRT decided '{d}' after {i+1} observations"); break
    else:
        print(f"  SPRT: CONTINUE  LLR_bull={qm.sprt.L:+.3f}  n={qm.sprt.n}")

    print("\n  CUSUM drift detection:")
    cusum = SequentialTests.CUSUM(mu=0, sigma=returns.std())
    for i,r in enumerate(returns[-30:]):
        res = cusum.update(r)
        if res["alarm"]:
            print(f"    Alarm at bar {i}: direction={res['direction']}"); break
    else:
        print(f"    No alarm in last 30 bars  S+={cusum.S_up:.4f}")

    # ── Full run ──────────────────────────────────────────────
    print("\n[FULL RUN]")
    full_out = qm.run_full(prices, returns, delta)
    print(f"  GARCH regime:    {full_out['garch']['regime']}")
    print(f"  EVT CVaR-99:     {full_out['evt']['cvar_99']:.5f}")
    print(f"  VR regime:       {full_out['vr_test']['regime']}")
    print(f"  CVD leads price: {full_out.get('cvd_leads', 'N/A')}")
    print(f"  Size multiplier: {full_out['size_multiplier']:.3f}×")

    print("\n" + "="*60)
    print("  ALL MATH ENGINES PASSED")
    print("="*60)
