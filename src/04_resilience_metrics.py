"""Step 4 - Dimensions of fraud resilience beyond the average post-shock gap.

(i)  Amplification: sensitivity of an entity's fraud intensity to the common
     (leave-one-out bank) fraud factor, in first differences.
(ii) Persistence / recovery: AR(1) coefficient with entity & month effects
     (LSDV, and Anderson-Hsiao IV to remove Nickell bias) and implied half-life.
(iii) Variance ratio VR(3) of month-demeaned log fraud intensity
     (1 = random walk / permanent shocks, <1 = mean reversion).
(iv) Volatility: s.d. of monthly changes.

Persistence/volatility metrics are computed on 'fraud-active' institutions
(>= 5 reported events per month on average) because log-intensity of
institutions that essentially report no fraud is pure count noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402
import config as C  # noqa: E402

MIN_EVENTS = 5.0


def amplification(df, drop_step=False, B=5000):
    d = df.copy()
    banks = d[d.fintech == 0]
    # leave-one-out mean of y among banks, by month
    s_sum = banks.groupby("ym").y.sum()
    s_n = banks.groupby("ym").y.count()
    d["S"] = d.ym.map(s_sum) - np.where(d.fintech == 0, d.y, 0.0)
    d["S"] = d["S"] / (d.ym.map(s_n) - (d.fintech == 0).astype(int))
    d = d.sort_values(["entity_id", "ym"])
    d["dy"] = d.groupby("entity_id").y.diff()
    d["dS"] = d.groupby("entity_id").S.diff()
    d = d.dropna(subset=["dy", "dS"])
    if drop_step:
        d = d[d.ym != U.ymi(C.EVENT_DATE)]
    e, ne = U._codes(d.entity_id)

    def slopes(flag):
        f = flag[e].astype(float)
        X = np.column_stack([d.dS.to_numpy(), d.dS.to_numpy() * f])
        Z = np.column_stack([d.dy.to_numpy(), X])
        # entity demeaning
        ce = np.bincount(e, minlength=ne)
        Zt = Z - np.column_stack([(np.bincount(e, weights=Z[:, j], minlength=ne) / ce)[e] for j in range(Z.shape[1])])
        b = np.linalg.lstsq(Zt[:, 1:], Zt[:, 0], rcond=None)[0]
        return b[0], b[0] + b[1]

    ents = d.drop_duplicates("entity_id")
    flag = np.zeros(ne)
    flag[U._codes(d.entity_id)[0][d.fintech.to_numpy() == 1]] = 1
    bB, bF = slopes(flag)
    rng = np.random.default_rng(C.SEED)
    nF = int(flag.sum())
    dr = []
    for _ in range(B):
        fl = np.zeros(ne)
        fl[rng.choice(ne, nF, replace=False)] = 1
        a, b = slopes(fl)
        dr.append(b - a)
    p = (1 + np.sum(np.abs(dr) >= abs(bF - bB))) / (B + 1)
    return bB, bF, p


def lsdv_ar(df, B=3000):
    d = df.copy().sort_values(["entity_id", "ym"])
    d["yl1"] = d.groupby("entity_id").y.shift(1)
    d["yl2"] = d.groupby("entity_id").y.shift(2)
    d = d.dropna(subset=["yl1"]).copy()
    d["rho_b"] = d.yl1 * (1 - d.fintech)
    d["rho_f"] = d.yl1 * d.fintech
    m = U.TWFE(d, "y", ["rho_b", "rho_f"])
    out = {"lsdv_b": m.coef("rho_b")[0], "lsdv_f": m.coef("rho_f")[0],
           "se_b": m.coef("rho_b")[1], "se_f": m.coef("rho_f")[1]}
    # permutation of group labels among eligible entities
    rng = np.random.default_rng(C.SEED)
    ents = d.entity_id.unique()
    nF = d[d.fintech == 1].entity_id.nunique()
    diffs = []
    for _ in range(B):
        f = set(rng.choice(ents, nF, replace=False))
        dd = d.assign(fl=d.entity_id.isin(f).astype(float))
        dd["rho_b"] = dd.yl1 * (1 - dd.fl)
        dd["rho_f"] = dd.yl1 * dd.fl
        mm = U.TWFE(dd, "y", ["rho_b", "rho_f"])
        diffs.append(mm.b[1] - mm.b[0])
    out["p_perm"] = (1 + np.sum(np.abs(diffs) >= abs(out["lsdv_f"] - out["lsdv_b"]))) / (B + 1)
    # Anderson-Hsiao: dy_t on dy_{t-1}, instrument y_{t-2}
    a = d.dropna(subset=["yl2"]).copy()
    a["dy"] = a.y - a.yl1
    a["dyl1"] = a.yl1 - a.yl2
    a["ent"] = a.entity_id
    for g, nm in [(0, "b"), (1, "f")]:
        a[f"dyl1_{nm}"] = a.dyl1 * (a.fintech == g)
        a[f"z_{nm}"] = a.yl2 * (a.fintech == g)
    mdum = pd.get_dummies(a.ym, drop_first=True, dtype=float, prefix="m")
    exog = mdum
    iv = IV2SLS(a["dy"], exog, a[["dyl1_b", "dyl1_f"]], a[["z_b", "z_f"]]).fit(
        cov_type="clustered", clusters=a.entity_id)
    out["ah_b"], out["ah_f"] = iv.params["dyl1_b"], iv.params["dyl1_f"]
    out["ah_se_b"], out["ah_se_f"] = iv.std_errors["dyl1_b"], iv.std_errors["dyl1_f"]
    return out


def entity_metrics(df):
    d = df.copy()
    mm = d.groupby("ym").y.transform("mean")
    d["z"] = d.y - mm
    rows = []
    for eid, g in d.sort_values("ym").groupby("entity_id"):
        z = g.z.to_numpy()
        dz1 = z[1:] - z[:-1]
        dz3 = z[3:] - z[:-3]
        vr = np.var(dz3, ddof=1) / (3 * np.var(dz1, ddof=1)) if np.var(dz1) > 0 else np.nan
        rows.append(dict(entity_id=eid, group=g.group.iloc[0], vol=np.std(dz1, ddof=1), vr3=vr,
                         mean_events=g.fraud.mean(),
                         ac1=np.corrcoef(dz1[1:], dz1[:-1])[0, 1] if len(dz1) > 3 else np.nan))
    return pd.DataFrame(rows)


def perm_mean_diff(vals, flag, B=10000):
    rng = np.random.default_rng(C.SEED)
    vals = np.asarray(vals, float)
    flag = np.asarray(flag, bool)
    obs = vals[flag].mean() - vals[~flag].mean()
    dr = []
    for _ in range(B):
        f = np.zeros(len(vals), bool)
        f[rng.choice(len(vals), flag.sum(), replace=False)] = True
        dr.append(vals[f].mean() - vals[~f].mean())
    return obs, (1 + np.sum(np.abs(dr) >= abs(obs))) / (B + 1)


def main():
    df = U.load_panel()
    elig = df.groupby("entity_id").fraud.mean()
    elig = elig[elig >= MIN_EVENTS].index
    act = df[df.entity_id.isin(elig)].copy()
    print("eligible entities:", act.entity_id.nunique(), " fintech:", act[act.fintech == 1].entity_id.nunique(),
          act[act.fintech == 1].entity.unique())

    rows = []
    # amplification (all entities; and ex-step month; and eligible only)
    for lab, d_, ds in [("Amplification beta, all entities", df, False),
                        ("Amplification beta, excl. Jan-24 step", df, True),
                        ("Amplification beta, fraud-active entities", act, False)]:
        bB, bF, p = amplification(d_, drop_step=ds)
        rows.append([lab, bF, bB, bF - bB, p])
    # persistence
    ar = lsdv_ar(act)
    hl = lambda r: np.log(.5) / np.log(r) if 0 < r < 1 else np.nan
    rows.append(["AR(1) persistence, LSDV", ar["lsdv_f"], ar["lsdv_b"], ar["lsdv_f"] - ar["lsdv_b"], ar["p_perm"]])
    # Anderson-Hsiao IV estimates are computed (ar["ah_*"]) but not tabulated: with T=23 and two
    # active fintechs the second-lag instrument is too weak (see paper, Section 5.3).
    rows.append(["Half-life of shocks, months (LSDV)", hl(ar["lsdv_f"]), hl(ar["lsdv_b"]), np.nan, np.nan])
    # entity metrics
    em = entity_metrics(act)
    em.to_csv(C.TABLES / "entity_resilience_metrics_raw.csv", index=False)
    flag = (em.group == "Fintech").to_numpy()
    for v, lab in [("vr3", "Variance ratio VR(3)"), ("vol", "Volatility: s.d. of monthly change"),
                   ("ac1", "Autocorrelation of monthly changes")]:
        x = em[v].to_numpy()
        ok = ~np.isnan(x)
        obs, p = perm_mean_diff(x[ok], flag[ok])
        rows.append([lab, x[ok & flag].mean(), x[ok & ~flag].mean(), obs, p])
    R = pd.DataFrame(rows, columns=["Metric", "Fintech", "Banks", "Diff.", "p_perm"])
    R.to_csv(C.TABLES / "tab6_resilience_raw.csv", index=False)
    print(R.round(3).to_string())
    print({k: round(v, 3) for k, v in ar.items()})
    print(em.assign(g=em.group).round(3).sort_values("group").to_string())

    T = R.copy()
    for c in ["Fintech", "Banks", "Diff."]:
        T[c] = T[c].map(lambda x: "" if np.isnan(x) else f"{x:.3f}")
    T["p (perm.)"] = T.p_perm.map(lambda x: "" if np.isnan(x) else f"{x:.3f}")
    T = T.drop(columns="p_perm")
    nF = act[act.fintech == 1].entity_id.nunique()
    nB = act[act.fintech == 0].entity_id.nunique()
    U.save_table(T, "tab6_resilience",
                 caption="Dimensions of fraud resilience: amplification, persistence and stability",
                 label="tab:res", col_format="lrrrr",
                 notes=f"Fraud-active institutions only for persistence and stability metrics ({nF} fintechs, "
                       f"{nB} banks with at least 5 reported events per month); amplification rows use all "
                       "entities in the first line. y = ln(1 + fraud per 100k clients). Amplification: slope of "
                       "the monthly change in an entity's y on the monthly change in the leave-one-out mean of banks. "
                       "Persistence: AR(1) with entity and month effects (LSDV; Nickell bias is of similar size in both groups, "
                       "so the difference is informative). VR(3) is computed on month-demeaned y. "
                       "p-values: randomization inference over entity labels.")


if __name__ == "__main__":
    main()
