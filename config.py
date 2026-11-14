"""Central configuration for the replication package.

Paper: "Agile or Fragile? Fraud Resilience of Fintech Companies and Traditional
Banks in Colombia".

All scripts import this module so that paths, sample windows and the
classification of institutions are defined in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Raw workbook (NOT distributed in the repository, see data/README.md).
# Override with the environment variable FRAUD_RAW_XLSX if stored elsewhere.
RAW_XLSX = Path(os.environ.get("FRAUD_RAW_XLSX", DATA_RAW / "Base Final Fraud.xlsx"))
RAW_SHEET = "FINAL"

PANEL_CSV = DATA_PROC / "panel_monthly_2019_2024.csv"

# --------------------------------------------------------------------------
# Sample windows
# --------------------------------------------------------------------------
# Fraud reporting changes taxonomy in 2023m1 (all events re-coded under the
# ACTIVO/PASIVO/NO APLICA scheme) and 2025 has no fraud records yet.  The
# regime-consistent estimation window is therefore 2023m1-2024m12.
FULL_START, FULL_END = (2019, 1), (2024, 12)
MAIN_START, MAIN_END = (2023, 1), (2024, 12)
EVENT_DATE = (2024, 1)        # system-wide escalation of reported fraud
PLACEBO_DATE = (2023, 7)      # in-time placebo inside the pre-event year

# --------------------------------------------------------------------------
# Institution classification
# --------------------------------------------------------------------------
# TIPOENTIDAD == 128 -> SEDPE (Sociedades Especializadas en Depositos y Pagos
# Electronicos, Law 1735/2014), the licensed fintech payment/deposit entities.
FINTECH_TYPE = 128

# Codes (TIPOENTIDAD_CODIGOENTIDAD) of entities that appear in the workbook
# with no name and no usable customer / fraud information.  They are dropped.
UNIDENTIFIED = {"1_0", "1_4", "1_5", "1_8", "1_10", "1_14", "1_15", "1_21",
                "1_31", "1_33", "1_37", "1_38", "1_50", "1_64", "1_66",
                "1_68", "1_999", "128_0", "128_6", "128_999"}

# Wholesale / no-retail-customer entities (customer denominator is zero).
NO_RETAIL_BASE = {"1_9", "1_59"}  # Citibank-Colombia, Santander de Negocios

MICROFINANCE = {"1_52", "1_53", "1_60", "1_62"}   # Bancamia, Banco W, Mundo Mujer, Mibanco
DIGITAL_BANK = {"1_65"}                            # Lulo Bank (neobank licence)
LARGE_BANK_MIN_CLIENTS = 1_000_000   # median retail clients in 2023-24 -> "large bank"

# --------------------------------------------------------------------------
# Inference settings
# --------------------------------------------------------------------------
SEED = 20241113
N_BOOT = 4999       # wild cluster bootstrap draws
N_PERM = 20000      # randomization-inference draws
