"""Shared helpers: data loading, two-way fixed-effects OLS with cluster-robust
(CR1) inference, wild cluster bootstrap (WCR, Webb weights), randomization
inference and LaTeX table writers.

With only five fintech entities (treated clusters) conventional clustered
standard errors over-reject, so every headline result is accompanied by
wild-cluster-bootstrap and permutation p-values (Cameron, Gelbach & Miller,
2008; MacKinnon & Webb, 2017; Conley & Taber, 2011).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def ymi(ym: Tuple[int, int]) -> int:
    return ym[0] * 100 + ym[1]


def month_index(ym: pd.Series) -> pd.Series:
    """Consecutive month counter (2019m1 = 0)."""
    return (ym // 100 - 2019) * 12 + (ym % 100 - 1)


def load_panel(start=C.MAIN_START, end=C.MAIN_END, drop_no_base=True) -> pd.DataFrame:
    df = pd.read_csv(C.PANEL_CSV, parse_dates=["date"])
    df = df[(df.ym >= ymi(start)) & (df.ym <= ymi(end))].copy()
    if drop_no_base:
        df = df[(df.no_retail_base == 0) & (df.customers > 0)].copy()
    df["t"] = month_index(df.ym)
    df["post"] = (df.ym >= ymi(C.EVENT_DATE)).astype(int)
    df["k"] = df["t"] - int(month_index(pd.Series([ymi(C.EVENT_DATE)])).iloc[0])  # event time
    df["rate"] = df["fraud"] / df["customers"] * 1e5
    df["y"] = np.log1p(df["rate"])
    df["ln_cust"] = np.log(df["customers"])
    df["asinh_rate"] = np.arcsinh(df["rate"])
    df = df.sort_values(["entity_id", "t"]).reset_index(drop=True)
    df["ln_cust_l1"] = df.groupby("entity_id")["ln_cust"].shift(1)
    df["ln_cust_l1"] = df["ln_cust_l1"].fillna(df["ln_cust"])
    return df


# ---------------------------------------------------------------------------
# Two-way fixed effects machinery
# ---------------------------------------------------------------------------

def _codes(s) -> Tuple[np.ndarray, int]:
    c, u = pd.factorize(pd.Series(s))
    return c.astype(int), len(u)


def demean(Y: np.ndarray, e: np.ndarray, t: np.ndarray, ne: int, nt: int,
           tol: float = 1e-11, maxit: int = 5000) -> np.ndarray:
    """Alternating-projection removal of entity and time effects (unbalanced OK)."""
    Y = np.array(Y, dtype=float, copy=True)
    if Y.ndim == 1:
        Y = Y[:, None]
    ce = np.bincount(e, minlength=ne).astype(float)
    ct = np.bincount(t, minlength=nt).astype(float)
    for _ in range(maxit):
        old = Y.copy()
        for j in range(Y.shape[1]):
            Y[:, j] -= (np.bincount(e, weights=Y[:, j], minlength=ne) / ce)[e]
            Y[:, j] -= (np.bincount(t, weights=Y[:, j], minlength=nt) / ct)[t]
        if np.max(np.abs(Y - old)) < tol:
            break
    return Y


def cr1_vcov(X: np.ndarray, u: np.ndarray, cl: np.ndarray, nG: int, n_extra: int = 0) -> np.ndarray:
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for g in range(nG):
        m = cl == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    adj = nG / (nG - 1) * (n - 1) / (n - k - n_extra)
    return adj * XtXi @ meat @ XtXi


class TWFE:
    """Two-way FE regression y ~ X | entity + time, cluster = entity."""

    def __init__(self, df: pd.DataFrame, yvar: str, xvars: Sequence[str],
                 cluster: str = "entity_id", time: str = "ym"):
        self.df = df
        self.yvar, self.xvars = yvar, list(xvars)
        self.e, self.ne = _codes(df["entity_id"])
        self.t, self.nt = _codes(df[time])
        self.cl, self.nG = _codes(df[cluster])
        Z = df[[yvar] + self.xvars].to_numpy(float)
        Zt = demean(Z, self.e, self.t, self.ne, self.nt)
        self.yt, self.Xt = Zt[:, 0], Zt[:, 1:]
        self.fit()

    def fit(self):
        X, y = self.Xt, self.yt
        self.b = np.linalg.lstsq(X, y, rcond=None)[0]
        self.u = y - X @ self.b
        self.V = cr1_vcov(X, self.u, self.cl, self.nG, n_extra=self.nt - 1)
        self.se = np.sqrt(np.diag(self.V))
        self.tval = self.b / self.se
        self.p = 2 * stats.t.sf(np.abs(self.tval), self.nG - 1)
        self.n = len(y)

    def coef(self, name: str):
        i = self.xvars.index(name)
        return self.b[i], self.se[i], self.tval[i], self.p[i]

    # -- wild cluster bootstrap, null imposed (WCR) -------------------------
    def wcr_pvalue(self, name: str, B: int = C.N_BOOT, seed: int = C.SEED,
                   weights: str = "webb") -> float:
        rng = np.random.default_rng(seed)
        j = self.xvars.index(name)
        X, y = self.Xt, self.yt
        keep = [i for i in range(X.shape[1]) if i != j]
        if keep:
            gam = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
            fit0 = X[:, keep] @ gam
        else:
            fit0 = np.zeros_like(y)
        u0 = y - fit0
        t_obs = self.tval[j]
        # pseudo-inverse reused across draws
        XtXi = np.linalg.pinv(X.T @ X)
        tstar = np.empty(B)
        if weights == "webb":
            vals = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
            W = rng.choice(vals, size=(B, self.nG))
        else:
            W = rng.choice([-1.0, 1.0], size=(B, self.nG))
        for b in range(B):
            ystar = fit0 + u0 * W[b, self.cl]
            ystar = demean(ystar, self.e, self.t, self.ne, self.nt)[:, 0]
            bs = XtXi @ (X.T @ ystar)
            us = ystar - X @ bs
            Vs = cr1_vcov(X, us, self.cl, self.nG, n_extra=self.nt - 1)
            tstar[b] = bs[j] / np.sqrt(Vs[j, j])
        return float(np.mean(np.abs(tstar) >= np.abs(t_obs)))


def did_design(df: pd.DataFrame, treat: str = "fintech", post: str = "post") -> pd.DataFrame:
    d = df.copy()
    d["did"] = d[treat] * d[post]
    return d


def permutation_did(df: pd.DataFrame, yvar: str, controls: Sequence[str] = (),
                    B: int = C.N_PERM, seed: int = C.SEED,
                    post: str = "post", treat: str = "fintech") -> Tuple[float, np.ndarray, float]:
    """Randomization inference for the DiD coefficient.

    Re-draws which entities are 'fintech' (preserving the number of treated)
    and recomputes the FE-DiD coefficient; returns (beta, null draws, two-sided p).
    """
    rng = np.random.default_rng(seed)
    e, ne = _codes(df["entity_id"])
    t, nt = _codes(df["ym"])
    Z = df[[yvar] + list(controls)].to_numpy(float)
    Zt = demean(Z, e, t, ne, nt)
    yt, Ct = Zt[:, 0], Zt[:, 1:]
    postv = df[post].to_numpy(float)
    ents = np.unique(e)
    n_f = int(df.loc[df[treat] == 1, "entity_id"].nunique())
    obs_flag = np.zeros(ne)
    obs_flag[np.unique(e[df[treat].to_numpy() == 1])] = 1

    def beta_for(flag):
        d = (flag[e] * postv)[:, None]
        dt = demean(d, e, t, ne, nt)
        X = np.hstack([dt, Ct]) if Ct.size else dt
        return np.linalg.lstsq(X, yt, rcond=None)[0][0]

    b_obs = beta_for(obs_flag)
    draws = np.empty(B)
    for i in range(B):
        flag = np.zeros(ne)
        flag[rng.choice(ents, size=n_f, replace=False)] = 1
        draws[i] = beta_for(flag)
    p = (1 + np.sum(np.abs(draws) >= np.abs(b_obs) - 1e-12)) / (B + 1)
    return b_obs, draws, float(p)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def fmt(x, d=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{d}f}"


def save_table(df: pd.DataFrame, name: str, caption: str = "", label: str = "",
               notes: str = "", index: bool = False, col_format: Optional[str] = None,
               escape: bool = True, font: str = "\\small") -> None:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / f"{name}.csv", index=index)
    body = df.to_latex(index=index, escape=escape, column_format=col_format)
    # booktabs-style rules
    body = body.replace("\\toprule", "\\toprule").replace("\\hline", "")
    tex = ["\\begin{table}[!htbp]\\centering", font]
    if caption:
        tex.append(f"\\caption{{{caption}}}")
    if label:
        tex.append(f"\\label{{{label}}}")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append(body.strip())
    tex.append("}")
    if notes:
        tex.append(f"\\par\\vspace{{2pt}}\\parbox{{\\textwidth}}{{\\footnotesize {notes}}}")
    tex.append("\\end{table}")
    (C.TABLES / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
