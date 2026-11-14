"""Step 3 - Baseline gap, difference-in-differences around the Jan-2024 fraud
escalation, event study and recovery slope.

Outcome: y = ln(1 + fraud events per 100,000 retail clients), monthly.
Model:   y_it = a_i + d_t + b * (Fintech_i x Post_t) + g' X_{i,t-1} + e_it
Inference: CR1 clustered, wild cluster bootstrap (WCR, Webb), randomization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402
import config as C  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
ACTIVE_FINTECH = {"128_3", "128_5"}      # Aval Soluciones Digitales, Coink: material fraud reporting


def run_did(df, yvar="y", controls=(), B=C.N_BOOT, P=C.N_PERM, label=""):
    d = U.did_design(df)
    xs = ["did"] + list(controls)
    m = U.TWFE(d, yvar, xs)
    b, se, t, p = m.coef("did")
    p_w = m.wcr_pvalue("did", B=B)
    _, draws, p_perm = U.permutation_did(d, yvar, controls, B=P)
    return dict(label=label, beta=b, se=se, p_cr1=p, p_wcr=p_w, p_perm=p_perm,
                n=m.n, n_ent=m.ne, n_fin=int(d[d.fintech == 1].entity_id.nunique())), draws


def ppml_did(df, offset="customers"):
    d = df.copy()
    tot = d.groupby("entity_id").fraud.transform("sum")
    d = d[tot > 0].copy()
    d["did"] = d.fintech * d.post
    X = pd.concat([d[["did"]],
                   pd.get_dummies(d.entity_id, drop_first=True, dtype=float),
                   pd.get_dummies(d.ym, drop_first=True, dtype=float, prefix="m")], axis=1)
    X = sm.add_constant(X)
    mod = sm.GLM(d.fraud, X, family=sm.families.Poisson(), offset=np.log(d[offset]))
    r = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.entity_id)[0]})
    return r.params["did"], r.bse["did"], r.pvalues["did"], int(d.entity_id.nunique())


def baseline_gap(df, label_groups):
    """Cross-sectional gap with month FE (no entity FE)."""
    rows = []
    for name, sub in label_groups.items():
        d = sub.copy()
        X = pd.concat([d[["fintech"]], pd.get_dummies(d.ym, drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X)
        r = sm.OLS(d["y"], X).fit(cov_type="cluster",
                                   cov_kwds={"groups": pd.factorize(d.entity_id)[0]})
        # permutation (entity-level re-labelling)
        rng = np.random.default_rng(C.SEED)
        ents = d.entity_id.unique()
        nf = d[d.fintech == 1].entity_id.nunique()
        ybar = d.groupby("entity_id").y.mean()
        obs = ybar[d[d.fintech == 1].entity_id.unique()].mean() - ybar[[e for e in ents if e not in d[d.fintech == 1].entity_id.unique()]].mean()
        dr = []
        for _ in range(10000):
            f = rng.choice(ents, nf, replace=False)
            dr.append(ybar[f].mean() - ybar[[e for e in ents if e not in f]].mean())
        pp = (1 + np.sum(np.abs(dr) >= abs(obs))) / 10001
        rows.append(dict(sample=name, gap=r.params["fintech"], se=r.bse["fintech"],
                         p_cr1=r.pvalues["fintech"], p_perm=pp, n=int(r.nobs),
                         n_ent=d.entity_id.nunique(), n_fin=nf))
    return pd.DataFrame(rows)


def event_study(df, treat="fintech", lo=-12, hi=11, ref=-1):
    d = df.copy()
    ks = [k for k in range(lo, hi + 1) if k != ref]
    for k in ks:
        d[f"es_{k}"] = ((d.k == k) & (d[treat] == 1)).astype(float)
    m = U.TWFE(d, "y", [f"es_{k}" for k in ks])
    out = []
    crit = stats.t.ppf(.975, m.nG - 1)
    for k in ks:
        b, se, t, p = m.coef(f"es_{k}")
        out.append((k, b, se, b - crit * se, b + crit * se, p))
    out.append((ref, 0.0, 0.0, 0.0, 0.0, np.nan))
    return pd.DataFrame(out, columns=["k", "b", "se", "lo", "hi", "p"]).sort_values("k")


def main():
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df = U.load_panel()
    df["active_fintech"] = df.entity_id.isin(ACTIVE_FINTECH).astype(int)
    banks = df[df.fintech == 0]
    fin = df[df.fintech == 1]
    print("N entities:", df.entity_id.nunique(), "fintech", fin.entity_id.nunique())

    # ---------------- A. baseline levels -----------------------------------
    groups = {
        "All fintech vs all banks": df,
        "All fintech vs microfinance banks": pd.concat([fin, banks[banks.entity_type == "Microfinance bank"]]),
        "All fintech vs large banks": pd.concat([fin, banks[banks.entity_type == "Large bank"]]),
        "Active fintech vs all banks": pd.concat([fin[fin.entity_id.isin(ACTIVE_FINTECH)], banks]),
    }
    gap = baseline_gap(df, groups)
    gap.to_csv(C.TABLES / "tab3_baseline_gap_raw.csv", index=False)
    print(gap.round(3).to_string())

    # ---------------- B. DiD ---------------------------------------------------
    specs = [
        ("(1) All fintech vs banks", df, [], "y"),
        ("(2) + lagged ln clients", df, ["ln_cust_l1"], "y"),
        ("(3) vs microfinance banks", pd.concat([fin, banks[banks.entity_type == "Microfinance bank"]]), ["ln_cust_l1"], "y"),
        ("(4) vs large banks", pd.concat([fin, banks[banks.entity_type == "Large bank"]]), ["ln_cust_l1"], "y"),
        ("(5) Active fintech only", pd.concat([fin[fin.entity_id.isin(ACTIVE_FINTECH)], banks]), ["ln_cust_l1"], "y"),
    ]
    res, draws_main = [], None
    for lab, d, ctr, yv in specs:
        r, dr = run_did(d, yv, ctr, label=lab)
        res.append(r)
        if lab.startswith("(2)"):
            draws_main = dr
        print(lab, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    pb, pse, pp, pn = ppml_did(df)
    res.append(dict(label="(6) PPML (exposure = clients)", beta=pb, se=pse, p_cr1=pp, p_wcr=np.nan,
                    p_perm=np.nan, n=int((df.groupby('entity_id').fraud.transform('sum') > 0).sum()),
                    n_ent=pn, n_fin=int(df[(df.fintech == 1)].groupby('entity_id').fraud.sum().gt(0).sum())))
    R = pd.DataFrame(res)
    R.to_csv(C.TABLES / "tab4_did_raw.csv", index=False)

    # permutation distribution figure (spec 2)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.hist(draws_main, bins=60, color="#9AA9BD", edgecolor="white", lw=.3)
    b_obs = R.loc[1, "beta"]
    ax.axvline(b_obs, color="#C2410C", lw=1.5)
    ax.text(b_obs, ax.get_ylim()[1] * .92, f"  observed = {b_obs:.2f}", color="#C2410C", fontsize=7.5, ha="left" if b_obs < 0 else "right")
    ax.set_xlabel("Placebo DiD coefficient (random fintech labels)")
    ax.set_ylabel("Frequency")
    fig.savefig(C.FIGURES / "fig3_permutation.pdf"); fig.savefig(C.FIGURES / "fig3_permutation.png", dpi=300)
    plt.close(fig)

    # ---------------- C. event study ---------------------------------------
    es_all = event_study(df)
    es_act = event_study(df[(df.fintech == 0) | df.entity_id.isin(ACTIVE_FINTECH)])
    es_all.to_csv(C.TABLES / "eventstudy_all_raw.csv", index=False)
    es_act.to_csv(C.TABLES / "eventstudy_active_raw.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for ax, es, ttl in zip(axes, [es_all, es_act], ["(a) All five SEDPE fintechs", "(b) Active fintechs (Aval Soluciones, Coink)"]):
        ax.axhline(0, color="grey", lw=.8)
        ax.axvline(-.5, color="grey", ls="--", lw=.8)
        ax.fill_between(es.k, es.lo, es.hi, color="#F2C6B0", alpha=.6, lw=0)
        ax.plot(es.k, es.b, color="#C2410C", lw=1.3, marker="o", ms=3)
        ax.set_xlabel("Months relative to Jan-2024 (ref. = Dec-2023)")
        ax.set_title(ttl, fontsize=9, loc="left")
    axes[0].set_ylabel("Fintech minus bank gap\nin ln(1 + fraud per 100k clients)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig2_eventstudy.pdf"); fig.savefig(C.FIGURES / "fig2_eventstudy.png", dpi=300)
    plt.close(fig)

    # ---------------- D. pre-trend and recovery slope -------------------------
    d = df.copy()
    d["f_k_pre"] = d.fintech * np.where(d.k < 0, d.k + 12, 0)         # linear pre-trend, 0..11 (zero afterwards)
    d["did"] = d.fintech * d.post
    d["did_k"] = d.fintech * d.post * d.k                              # post-shock slope (months since shock)
    m = U.TWFE(d, "y", ["did", "did_k", "f_k_pre", "ln_cust_l1"])
    rows = []
    for nm, lab in [("did", "Fintech x Post (level shift at k=0)"),
                    ("did_k", "Fintech x Post x months since shock (recovery slope)"),
                    ("f_k_pre", "Fintech x pre-period linear trend")]:
        b, se, t, p = m.coef(nm)
        rows.append(dict(term=lab, beta=b, se=se, p_cr1=p, p_wcr=m.wcr_pvalue(nm, B=2999)))
    pd.DataFrame(rows).to_csv(C.TABLES / "tab_dynamics_raw.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())

    # ---------------- E. formatted LaTeX tables ---------------------------------
    def cell(b, se, p):
        return f"{b:.3f}{U.stars(p)}", f"({se:.3f})"
    g = gap.copy()
    t3 = pd.DataFrame({
        "Comparison": g["sample"],
        "Fintech gap": [f"{a:.3f}{U.stars(p)}" for a, p in zip(g.gap, g.p_perm)],
        "(CR1 s.e.)": [f"({s:.3f})" for s in g.se],
        "p (CR1)": g.p_cr1.map(lambda x: f"{x:.3f}"),
        "p (perm.)": g.p_perm.map(lambda x: f"{x:.3f}"),
        "Obs.": g.n, "Entities": g.n_ent, "Fintechs": g.n_fin})
    U.save_table(t3, "tab3_baseline_gap",
                 caption="Baseline gap in fraud intensity between fintechs and banks (month fixed effects)",
                 label="tab:gap", col_format="lrrrrrrr",
                 notes="Dependent variable: ln(1 + fraud events per 100,000 retail clients), 2023m1--2024m12. "
                       "Regressions include month fixed effects only. Stars follow the randomization-inference "
                       "p-value (* p<0.10, ** p<0.05, *** p<0.01).")
    t4 = pd.DataFrame({
        "Specification": R.label,
        "Fintech x Post": [f"{a:.3f}{U.stars(p if not np.isnan(p) else pc)}" for a, p, pc in zip(R.beta, R.p_perm, R.p_cr1)],
        "(CR1 s.e.)": [f"({s:.3f})" for s in R.se],
        "p (CR1)": R.p_cr1.map(lambda x: f"{x:.3f}"),
        "p (WCR)": R.p_wcr.map(lambda x: "" if np.isnan(x) else f"{x:.3f}"),
        "p (perm.)": R.p_perm.map(lambda x: "" if np.isnan(x) else f"{x:.3f}"),
        "Entities": R.n_ent, "Fintechs": R.n_fin})
    U.save_table(t4, "tab4_did",
                 caption="Absorption of the January-2024 fraud escalation: difference-in-differences estimates",
                 label="tab:did", col_format="lrrrrrrr",
                 notes="Dependent variable: ln(1 + fraud events per 100,000 retail clients) in specs (1)--(5); "
                       "fraud event counts in spec (6) (Poisson pseudo-maximum likelihood, clients as exposure; "
                       "coefficient is a semi-elasticity). All models include entity and month fixed effects; "
                       "post = 2024m1 onward. Stars follow the permutation p-value (CR1 p-value in (6)). "
                       "WCR: wild cluster bootstrap-t with Webb weights, null imposed (4,999 draws). "
                       "Permutation: 20,000 random re-assignments of the fintech label across entities.")
    dd = pd.DataFrame(rows)
    t5 = pd.DataFrame({"Term": dd.term,
                       "Coefficient": [f"{a:.3f}" for a in dd.beta],
                       "(CR1 s.e.)": [f"({s:.3f})" for s in dd.se],
                       "p (CR1)": dd.p_cr1.map(lambda x: f"{x:.3f}"),
                       "p (WCR)": dd.p_wcr.map(lambda x: f"{x:.3f}")})
    U.save_table(t5, "tab5_dynamics",
                 caption="Pre-trend and post-shock recovery slope of the fintech--bank gap",
                 label="tab:dyn", col_format="lrrrr",
                 notes="Model: entity and month fixed effects plus lagged ln clients. The pre-trend term is a linear "
                       "trend in the fintech indicator over 2023 (months 0--11); the recovery slope is the change "
                       "per month in the fintech--bank gap after the shock (negative = convergence).")


if __name__ == "__main__":
    main()
