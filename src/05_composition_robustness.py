"""Step 5 - Fraud composition (product side) and robustness of the DiD estimate."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402
import config as C  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
ACTIVE = {"128_3", "128_5"}
B_ROB, P_ROB = 1999, 5000


def did_stats(d, yvar="y", controls=("ln_cust_l1",), label="", B=B_ROB, P=P_ROB, post="post"):
    d = d.copy()
    d["did"] = d.fintech * d[post]
    m = U.TWFE(d, yvar, ["did"] + list(controls))
    b, se, t, p = m.coef("did")
    _, _, pp = U.permutation_did(d, yvar, controls, B=P, post=post)
    return dict(Variant=label, beta=b, se=se, p_cr1=p, p_wcr=m.wcr_pvalue("did", B=B), p_perm=pp,
                N=m.n, entities=m.ne, fintechs=int(d[d.fintech == 1].entity_id.nunique()))


def composition(df):
    d = df.copy()
    rows = []
    for g, s in d.groupby("group"):
        tot = s.fraud.sum()
        ent = s.groupby("entity_id").agg(f=("fraud", "sum"), l=("fraud_liab", "sum"), a=("fraud_asset", "sum"),
                                          u=("unrecognised", "sum"), m=("misconduct", "sum"))
        ent = ent[ent.f > 0]
        rows.append(dict(Group=g,
                         **{"Fraud events": int(tot),
                            "Deposit/payment side (%)": 100 * s.fraud_liab.sum() / tot,
                            "Credit side (%)": 100 * s.fraud_asset.sum() / tot,
                            "Unclassified side (%)": 100 * s.fraud_na.sum() / tot,
                            "Unrecognised-transaction claims per fraud event": s.unrecognised.sum() / tot,
                            "Misconduct claims per fraud event": s.misconduct.sum() / tot,
                            "Mean entity share, deposit side (%)": 100 * (ent.l / ent.f).mean()}))
    return pd.DataFrame(rows)


def main():
    df = U.load_panel()
    banks = df[df.fintech == 0]

    # ---- composition ------------------------------------------------------
    comp = composition(df)
    comp.to_csv(C.TABLES / "tab7_composition_raw.csv", index=False)
    print(comp.round(2).T.to_string())
    ct = comp.set_index("Group").T
    ct.index.name = "Measure"
    ct = ct.reset_index()
    for c in ["Bank", "Fintech"]:
        ct[c] = [f"{float(v):,.0f}" if i == 0 else f"{float(v):.2f}" for i, v in enumerate(ct[c])]
    U.save_table(ct, "tab7_composition",
                 caption="Product-side composition of reported fraud, 2023m1--2024m12",
                 label="tab:comp", col_format="lrr",
                 notes="Pooled event counts. SEDPEs are licensed to take deposits and provide payments only, so "
                       "credit-side fraud is structurally absent for fintechs. Unrecognised-transaction and misconduct "
                       "claims are separate SFC categories not included in the fraud counts.")

    # ---- robustness of DiD ---------------------------------------------------
    rob = []
    rob.append(did_stats(df, label="Baseline (spec. 2)"))
    d = df.copy(); d["y_asinh"] = d.asinh_rate
    rob.append(did_stats(d, "y_asinh", label="asinh(rate) outcome"))
    d = df.copy(); d["y_liab"] = np.log1p(d.fraud_liab / d.customers * 1e5)
    rob.append(did_stats(d, "y_liab", label="Deposit-side fraud only (like-for-like products)"))
    d = df.copy(); d["y_liab"] = np.log1p(d.fraud_liab / d.customers * 1e5)
    rob.append(did_stats(d[(d.fintech == 0) | d.entity_id.isin(ACTIVE)], "y_liab",
                         label="Deposit-side fraud only, active fintechs only"))
    d = df.copy(); d["y_broad"] = np.log1p(d.fraud_broad / d.customers * 1e5)
    rob.append(did_stats(d, "y_broad", label="Broad measure: fraud + unrecognised-transaction claims"))
    d = df.copy(); d["y_cnt"] = np.log1p(d.fraud)
    rob.append(did_stats(d, "y_cnt", label="ln(1 + fraud count), size control only"))
    d = df.dropna(subset=["digital_txn"]).copy(); d["y_dig"] = np.log1p(d.fraud / d.digital_txn * 1e6)
    rob.append(did_stats(d, "y_dig", label="Per million digital-channel transactions"))
    rob.append(did_stats(df[~df.entity_id.isin(["1_7", "1_39"])], label="Excl. Bancolombia and Davivienda"))
    rob.append(did_stats(df[df.entity_type != "Digital bank"], label="Excl. Lulo Bank (digital bank)"))
    d = df.copy(); d.loc[d.entity_id == "1_65", "fintech"] = 1
    rob.append(did_stats(d, label="Lulo Bank re-classified as fintech"))
    d = df[df.entity_id.map(df.groupby("entity_id").fraud.mean()) >= 5].copy()
    rob.append(did_stats(d, label="Fraud-active institutions only (>=5 events/month)"))
    d = df.copy(); lo, hi = d.y.quantile([.05, .95]); d["y_w"] = d.y.clip(lo, hi)
    rob.append(did_stats(d, "y_w", label="Outcome winsorised at 5/95%"))
    rob.append(did_stats(df[~df.ym.isin([202401, 202402])], label="Donut: drop Jan-Feb 2024"))
    d = df.copy(); d["post"] = (d.ym >= 202404).astype(int)
    rob.append(did_stats(d, label="Late shock date (Apr-2024)"))
    # in-time placebo within 2023
    d = U.load_panel(start=(2023, 1), end=(2023, 12)); d["post_pl"] = (d.ym >= 202307).astype(int)
    rob.append(did_stats(d, label="In-time placebo: shock set to Jul-2023 (2023 only)", post="post_pl"))
    # leave-one-fintech-out
    for fid in sorted(df[df.fintech == 1].entity_id.unique()):
        nm = df.loc[df.entity_id == fid, "entity"].iloc[0]
        rob.append(did_stats(df[df.entity_id != fid], label=f"Leave out fintech: {nm}"))
    R = pd.DataFrame(rob)
    R.to_csv(C.TABLES / "tab8_robustness_raw.csv", index=False)
    print(R.round(3).to_string())

    T = pd.DataFrame({
        "Variant": R.Variant,
        "Fintech x Post": [f"{a:.3f}{U.stars(p)}" for a, p in zip(R.beta, R.p_perm)],
        "(CR1 s.e.)": [f"({s:.3f})" for s in R.se],
        "p (CR1)": R.p_cr1.map(lambda x: f"{x:.3f}"),
        "p (WCR)": R.p_wcr.map(lambda x: f"{x:.3f}"),
        "p (perm.)": R.p_perm.map(lambda x: f"{x:.3f}"),
        "Entities": R.entities, "Fintechs": R.fintechs})
    U.save_table(T, "tab8_robustness",
                 caption="Robustness of the post-escalation fintech--bank differential",
                 label="tab:rob", col_format="lrrrrrrr",
                 notes="Each row re-estimates the baseline DiD (entity and month fixed effects, lagged ln clients) "
                       "under the indicated change. Stars follow the permutation p-value (1,999 WCR draws; "
                       "5,000 permutations). The Jul-2023 placebo uses 2023 only.")

    # ---- forest plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    y = np.arange(len(R))[::-1]
    crit = 2.05
    ax.axvline(0, color="grey", lw=.8)
    ax.errorbar(R.beta, y, xerr=crit * R.se, fmt="o", color="#C2410C", ecolor="#E0A98F", ms=4, lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels([v if len(v) < 52 else v[:49] + "..." for v in R.Variant], fontsize=7)
    ax.set_xlabel("Fintech x Post coefficient (95% CR1 interval)")
    fig.savefig(C.FIGURES / "fig4_robustness.pdf"); fig.savefig(C.FIGURES / "fig4_robustness.png", dpi=300)
    plt.close(fig)

    # ---- trajectories of the two groups (levels) --------------------------------
    f = df.copy()
    f["grp"] = np.where(f.group == "Fintech", np.where(f.entity_id.isin(ACTIVE), "Active fintechs", "Silent fintechs"),
                        "Banks")
    g = f.groupby(["grp", "date"]).y.mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for k, c in [("Banks", "#1F3A5F"), ("Active fintechs", "#C2410C"), ("Silent fintechs", "#D9A441")]:
        s = g[g.grp == k]
        ax.plot(s.date, s.y, color=c, lw=1.4, label=k)
    ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls="--", lw=.8)
    ax.set_ylabel("Mean ln(1 + fraud per 100k clients)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(frameon=False, fontsize=7, loc="center left")
    fig.savefig(C.FIGURES / "fig5_means.pdf"); fig.savefig(C.FIGURES / "fig5_means.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
