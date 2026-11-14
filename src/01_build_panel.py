"""Step 1 - Build the entity-month analysis panel.

Reads the harmonised workbook ("Base Final Fraud.xlsx", sheet FINAL) compiled
from the Superintendencia Financiera de Colombia (SFC) fraud/claims reports,
customer statistics and channel-transaction reports, cleans it and writes
``data/processed/panel_monthly_2019_2024.csv``.

Cleaning decisions (all documented in data/README.md):
  * Entity names were corrupted by a find/replace in the workbook ("na" -> "0",
    e.g. "Ba0grario" = "Banagrario"); they are repaired here.
  * Entities without a name / customers / fraud records are dropped.
  * Fraud columns are parsed as SIDE_KIND_SUBTYPE.  ``total_fraud`` equals the
    sum of all ``*_Fraude_*`` columns (verified below).
  * Monthly digital-channel transactions (internet + mobile) exist only from
    2023m1 (before that they are semester snapshots) and are used only in
    robustness checks.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def fix_name(name: str) -> str:
    name = str(name).strip()
    if name in {"0", "nan"}:
        return ""
    return name.replace("0", "na")


def parse_fraud_cols(cols):
    """Return {col: (side, kind, subtype)} for the SIDE_KIND_SUBTYPE columns."""
    out = {}
    for c in cols:
        t = c.split("_")
        if t[0] in {"ACTIVO", "PASIVO", "NO APLICA"} and len(t) >= 3:
            side = {"ACTIVO": "asset", "PASIVO": "liab", "NO APLICA": "na"}[t[0]]
            k = t[1]
            kind = ("fraud" if k.startswith("Fraude") else
                    "misconduct" if k.startswith("Malas") else
                    "unrecognised" if k.startswith("No reconocido") else
                    "aml" if k.startswith("Lavado") else "other")
            out[c] = (side, kind, t[2])
    return out


def main() -> None:
    if not C.RAW_XLSX.exists():
        raise FileNotFoundError(
            f"Raw workbook not found at {C.RAW_XLSX}. See data/README.md; "
            "or run the remaining scripts directly on the processed panel.")
    raw = pd.read_excel(C.RAW_XLSX, sheet_name=C.RAW_SHEET)
    raw = raw.rename(columns={"YEAR": "year", "MES": "month"})
    raw["entity_id"] = raw["TIPOENTIDAD"].astype(str) + "_" + raw["CODIGOENTIDAD"].astype(str)

    # ---- sanity check on the fraud aggregate -----------------------------
    fcols = parse_fraud_cols(raw.columns)
    only_fraud = [c for c, v in fcols.items() if v[1] == "fraud"]
    chk = (raw[only_fraud].sum(axis=1) - raw["total_fraud"]).abs().max()
    print(f"max |sum(Fraude columns) - total_fraud| = {chk:.1f}")

    # ---- keep analysis window -------------------------------------------
    s = C.FULL_START[0] * 100 + C.FULL_START[1]
    e = C.FULL_END[0] * 100 + C.FULL_END[1]
    raw["ym"] = raw["year"] * 100 + raw["month"]
    df = raw[(raw.ym >= s) & (raw.ym <= e)].copy()

    # ---- names and classification ----------------------------------------
    names = (df.assign(nm=df["ENTIDAD"].map(fix_name))
               .loc[lambda x: x.nm != ""].groupby("entity_id")["nm"].first())
    df["entity"] = df["entity_id"].map(names)
    df = df[~df.entity_id.isin(C.UNIDENTIFIED) & df.entity.notna()].copy()
    df["fintech"] = (df["TIPOENTIDAD"] == C.FINTECH_TYPE).astype(int)

    # ---- fraud outcomes ---------------------------------------------------
    def agg(side=None, kind=None):
        cols = [c for c, (sd, kd, _) in fcols.items()
                if (side is None or sd == side) and (kind is None or kd == kind)]
        return df[cols].fillna(0).sum(axis=1)

    df["fraud"] = df["total_fraud"].fillna(0)
    df["fraud_liab"] = agg("liab", "fraud")        # deposit / payment-product side
    df["fraud_asset"] = agg("asset", "fraud")      # credit-product side
    df["fraud_na"] = agg("na", "fraud")
    df["unrecognised"] = agg(None, "unrecognised")
    df["misconduct"] = agg(None, "misconduct")
    df["fraud_broad"] = df["fraud"] + df["unrecognised"]
    # legacy typologies (2019-2022 only; collapse into SE after 2022)
    for k in ["ITF", "ATO", "APP", "SE"]:
        df[f"typ_{k}"] = df[f"fraud_{k}"].fillna(0)

    # ---- exposure variables ----------------------------------------------
    df["customers"] = df["clientes_naturales"].astype(float)
    df["customers_corp"] = df["clientes_corporativos"].astype(float)
    df["digital_txn"] = (df["Numero_transacciones_9"].fillna(0) +
                         df["Numero_transacciones_10"].fillna(0)).astype(float)
    # monthly digital transactions are only valid from 2023m1
    df.loc[df.ym < 202301, "digital_txn"] = np.nan
    df.loc[(df.ym >= 202301) & (df.digital_txn == 0), "digital_txn"] = np.nan

    df["has_credit"] = (df["cartera_TC_CR"].fillna(0) + df["cartera_microcredito"].fillna(0) > 0).astype(int)

    # ---- classification by type ------------------------------------------
    med_cust = df[df.ym >= 202301].groupby("entity_id")["customers"].median()

    def kind(r):
        if r.fintech == 1:
            return "Fintech (SEDPE)"
        if r.entity_id in C.MICROFINANCE:
            return "Microfinance bank"
        if r.entity_id in C.DIGITAL_BANK:
            return "Digital bank"
        if med_cust.get(r.entity_id, 0) >= C.LARGE_BANK_MIN_CLIENTS:
            return "Large bank"
        return "Other bank"

    ent = df.drop_duplicates("entity_id")[["entity_id", "fintech"]].copy()
    ent["entity_type"] = ent.apply(kind, axis=1)
    df = df.merge(ent[["entity_id", "entity_type"]], on="entity_id", how="left")
    df["group"] = np.where(df.fintech == 1, "Fintech", "Bank")
    df["no_retail_base"] = df.entity_id.isin(C.NO_RETAIL_BASE).astype(int)

    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    keep = ["entity_id", "entity", "group", "entity_type", "fintech", "year", "month", "ym", "date",
            "fraud", "fraud_liab", "fraud_asset", "fraud_na", "unrecognised", "misconduct",
            "fraud_broad", "typ_ITF", "typ_ATO", "typ_APP", "typ_SE",
            "customers", "customers_corp", "digital_txn", 
            "has_credit", "no_retail_base"]
    out = df[keep].sort_values(["entity_id", "ym"]).reset_index(drop=True)
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.PANEL_CSV, index=False)
    print(f"Wrote {C.PANEL_CSV}  rows={len(out)}  entities={out.entity_id.nunique()}")
    print(out.groupby(["group", "entity_type"]).entity_id.nunique())


if __name__ == "__main__":
    main()
