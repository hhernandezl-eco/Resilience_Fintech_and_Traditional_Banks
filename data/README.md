# Data

## Provenance
The analysis uses a harmonised entity-month workbook (`Base Final Fraud.xlsx`, sheet `FINAL`) compiled from
public regulatory reporting of the **Superintendencia Financiera de Colombia (SFC)**: fraud/claims event
reports by product side, customer statistics, and channel-transaction reports. The raw workbook is **not**
distributed here (size, and it is an intermediate compilation). To rebuild the processed panel from it, place it in
`data/raw/` (or set `FRAUD_RAW_XLSX`) and run `python run_all.py`.

`data/processed/panel_monthly_2019_2024.csv` is the analysis panel (entity x month, 2019m1-2024m12) and is
sufficient to reproduce every table and figure (the pipeline skips step 1 if the raw file is absent).

## Cleaning decisions
* Entity names in the workbook were corrupted by a find/replace ("na" -> "0", e.g. `Ba0grario` = Banagrario); repaired in `src/01_build_panel.py`.
* Unnamed entity codes with no clients or fraud records are dropped (`config.UNIDENTIFIED`).
* Citibank-Colombia and Banco Santander de Negocios have no retail-client base and are excluded from per-client models.
* `total_fraud` was verified to equal the sum of all `*_Fraude_*` columns.
* 2025 has no fraud records and is excluded. Fraud reporting changes taxonomy in 2023m1, and event counts step up again in 2024m1; the
  regime-consistent window is 2023m1-2024m12.
* Monthly digital-channel transactions (internet and mobile banking; codes 9 and 10 of the SFC transaction report, workbook variables `Numero_transacciones_9/10`) are valid only from 2023m1.

## Data dictionary (processed panel)
| Variable | Meaning |
|---|---|
| entity_id, entity | `TIPOENTIDAD_CODIGOENTIDAD` and repaired name |
| group | `Fintech` (SEDPE, TIPOENTIDAD 128) or `Bank` (TIPOENTIDAD 1) |
| entity_type | Fintech (SEDPE), Digital bank, Microfinance bank, Large bank (>=1m clients), Other bank |
| fraud | monthly count of events classified as fraud (all sides) |
| fraud_liab / fraud_asset / fraud_na | fraud events on deposit-payment side / credit side / not applicable |
| unrecognised, misconduct | claims of unrecognised transactions; bad-practice claims (not in `fraud`) |
| fraud_broad | fraud + unrecognised |
| typ_ITF, typ_ATO, typ_APP, typ_SE | legacy typologies (2019-2022 only) |
| customers, customers_corp | natural-person / corporate clients |
| digital_txn | monthly internet+mobile transactions (2023m1+ only) |
| has_credit, no_retail_base | flags |
