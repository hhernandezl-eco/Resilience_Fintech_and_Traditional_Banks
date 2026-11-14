"""Step 2 - Sample description, descriptive statistics and stylised-fact figures."""
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
COL = {"Fintech": "#C2410C", "Bank": "#1F3A5F", "Microfinance bank": "#2E8B57",
       "Large bank": "#1F3A5F", "Fintech (SEDPE)": "#C2410C", "Other bank": "#7A8BA3",
       "Digital bank": "#B8860B"}


def sample_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["entity_type", "entity"])
           .agg(months=("ym", "size"),
                clients_k=("customers", lambda s: s.median() / 1e3),
                fraud=("fraud", "sum"),
                rate=("rate", "mean"))
           .reset_index())
    order = ["Fintech (SEDPE)", "Digital bank", "Microfinance bank", "Large bank", "Other bank"]
    g["entity_type"] = pd.Categorical(g.entity_type, order, ordered=True)
    g = g.sort_values(["entity_type", "rate"], ascending=[True, False])
    g.columns = ["Type", "Institution", "Months", "Retail clients (000, median)",
                 "Fraud events 2023-24", "Mean fraud / 100k clients (monthly)"]
    g["Retail clients (000, median)"] = g["Retail clients (000, median)"].round(0).astype(int)
    g["Fraud events 2023-24"] = g["Fraud events 2023-24"].astype(int)
    g["Mean fraud / 100k clients (monthly)"] = g["Mean fraud / 100k clients (monthly)"].map(lambda x: f"{x:.2f}")
    g["Retail clients (000, median)"] = g["Retail clients (000, median)"].map(lambda x: f"{x:,}")
    g["Fraud events 2023-24"] = g["Fraud events 2023-24"].map(lambda x: f"{x:,}")
    return g


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["fraud_pc_liab"] = d["fraud_liab"] / d["customers"] * 1e5
    d["digital_pc"] = d["digital_txn"] / d["customers"]
    d["share_liab"] = np.where(d.fraud > 0, d.fraud_liab / d.fraud, np.nan)
    ent = d.groupby(["entity_id", "group"]).agg(
        rate=("rate", "mean"), fraud_m=("fraud", "mean"), cust=("customers", "median"),
        rate_liab=("fraud_pc_liab", "mean"), digital_pc=("digital_pc", "mean"),
        share_liab=("share_liab", "mean"),
        zero_share=("fraud", lambda s: (s == 0).mean())).reset_index()
    rows = []
    lab = {"rate": "Fraud events per 100k clients (monthly mean)",
           "fraud_m": "Fraud events (monthly mean)",
           "cust": "Retail clients (median)",
           "rate_liab": "Deposit-side fraud per 100k clients",
           "digital_pc": "Digital-channel transactions per client (monthly)",
           "zero_share": "Share of entity-months with zero reported fraud"}
    rng = np.random.default_rng(C.SEED)
    nF = (ent.group == "Fintech").sum()
    for v, l in lab.items():
        e = ent.dropna(subset=[v])
        f, b = e[e.group == "Fintech"][v], e[e.group == "Bank"][v]
        # permutation test on difference in entity means
        diff = f.mean() - b.mean()
        vals = e[v].to_numpy()
        draws = []
        for _ in range(10000):
            idx = rng.permutation(len(vals))
            draws.append(vals[idx[:len(f)]].mean() - vals[idx[len(f):]].mean())
        p = (1 + np.sum(np.abs(draws) >= abs(diff))) / (10000 + 1)
        rows.append([l, len(f), f.mean(), f.median(), len(b), b.mean(), b.median(), diff, p])
    out = pd.DataFrame(rows, columns=["Variable", "N fintech", "Fintech mean", "Fintech median",
                                      "N banks", "Bank mean", "Bank median", "Diff. in means",
                                      "Perm. p-value"])
    return out


def fig_timeseries(full: pd.DataFrame) -> None:
    f = full[(full.customers > 0) & (full.no_retail_base == 0)].copy()
    f["grp2"] = np.where(f.group == "Fintech", "Fintech (SEDPE)",
                         np.where(f.entity_type == "Microfinance bank", "Microfinance banks",
                                  "Other banks"))
    g = f.groupby(["grp2", "date"]).agg(fr=("fraud", "sum"), cu=("customers", "sum")).reset_index()
    g["rate"] = g.fr / g.cu * 1e5
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    colors = {"Fintech (SEDPE)": COL["Fintech"], "Microfinance banks": COL["Microfinance bank"],
              "Other banks": COL["Bank"]}
    for k, s in g.groupby("grp2"):
        s = s.sort_values("date")
        ax.plot(s.date, s.rate.clip(lower=0.003), color=colors[k], lw=1.3, label=k)
    ax.set_yscale("log")
    ax.axvline(pd.Timestamp("2023-01-01"), color="grey", ls=":", lw=.9)
    ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls="--", lw=.9)
    ax.text(pd.Timestamp("2023-01-15"), 0.0045, "Jan-23\nnew taxonomy", fontsize=6.5, color="grey")
    ax.text(pd.Timestamp("2024-01-15"), 0.0045, "Jan-24\nescalation", fontsize=6.5, color="grey")
    ax.set_ylabel("Reported fraud events\nper 100,000 clients (log)")
    ax.set_title("(a) Aggregate fraud intensity, 2019-2024", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[1]
    m = full[(full.ym >= 202301) & (full.ym <= 202412) & (full.customers > 0) & (full.no_retail_base == 0)]
    m = m.assign(rate=m.fraud / m.customers * 1e5)
    e = m.groupby(["entity_id", "group", "year"]).rate.mean().unstack()
    for grp, s in e.groupby(level="group"):
        ax.scatter(s[2023].clip(lower=.003), s[2024].clip(lower=.003), s=22, alpha=.85,
                   color=COL[grp], edgecolor="white", lw=.4, label=grp)
    lim = [.002, 400]
    ax.plot(lim, lim, color="grey", lw=.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("2023 mean rate"); ax.set_ylabel("2024 mean rate")
    ax.set_title("(b) Entity rates, 2024 vs 2023", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "fig1_timeseries.pdf"); fig.savefig(C.FIGURES / "fig1_timeseries.png", dpi=300)
    plt.close(fig)


def main() -> None:
    full = pd.read_csv(C.PANEL_CSV, parse_dates=["date"])
    df = U.load_panel()
    U.save_table(sample_table(df), "tab1_sample",
                 caption="Estimation sample: institutions, size and reported fraud (2023m1--2024m12)",
                 label="tab:sample", col_format="llrrrr",
                 notes="Fraud events are counts of events classified as fraud (``Fraude'') in the "
                       "SFC report. Clients are natural-person retail clients. Entities with no "
                       "retail-client base (Citibank-Colombia, Santander de Negocios) and unnamed "
                       "codes are excluded.")
    gs = group_summary(df)
    num = gs.columns[1:]
    gs_fmt = gs.copy()
    for c in num:
        if c in ("N fintech", "N banks"):
            gs_fmt[c] = gs_fmt[c].astype(int)
        else:
            gs_fmt[c] = gs_fmt[c].map(lambda x: f"{x:,.3f}" if abs(x) < 100 else f"{x:,.0f}")
    U.save_table(gs_fmt, "tab2_descriptives",
                 caption="Descriptive statistics by institution group (2023m1--2024m12, entity-level means)",
                 label="tab:desc", col_format="lrrrrrrrr",
                 notes="Entity-level averages over the estimation window. The last column reports the "
                       "randomization-inference p-value for the difference in means "
                       "(10,000 re-assignments of the fintech label across entities).")
    fig_timeseries(full)
    print(gs.round(3).to_string())
    print(sample_table(df).to_string())


if __name__ == "__main__":
    main()
