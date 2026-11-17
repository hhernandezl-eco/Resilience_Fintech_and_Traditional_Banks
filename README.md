# Agile or Fragile? Fraud resilience of fintechs and traditional banks in Colombia

**Design.** Monthly entity panel 2023m1-2024m12 (30 institutions: 5 licensed fintechs [SEDPE] and 25 banks).
Fintech vs bank differences in (i) fraud intensity, (ii) absorption of the January-2024 escalation (difference-in-differences,
event study, PPML), (iii) amplification, persistence and stability, (iv) product composition. Because only five fintechs exist,
all headline inference reports cluster-robust (CR1), wild-cluster-bootstrap and **randomization-inference** p-values.

## Reproduce
```bash
pip install -r requirements.txt
python run_all.py          # ~5 min; writes results/tables/*.tex|csv and results/figures/*.pdf|png
```
Python 3.9+ (tested 3.9). Seeds are fixed in `config.py`.

| Script | Output |
|---|---|
| `src/01_build_panel.py` | processed panel (needs raw workbook; skipped otherwise) |
| `src/02_descriptives.py` | Tables 1-2, Figure 1 |
| `src/03_main_did.py` | Tables 3-5 (gap, DiD, dynamics), Figures 2-3 |
| `src/04_resilience_metrics.py` | Table 6 (amplification, persistence, VR, volatility) |
| `src/05_composition_robustness.py` | Tables 7-8, Figures 4-5 |

See `data/README.md` for data provenance, cleaning rules and the data dictionary.
