# EE_Ref — Interactive Electrical Engineering Reference Scripts

Interactive [marimo](https://marimo.io) notebooks that accompany the *Electrical Engineering Reference* book. Each script contains visualizations, graphs, and interactive examples for a specific chapter of the book.

## Chapters Covered

| Script | Chapter |
|--------|---------|
| `01_power_engineering.py` | Chapter 1 — Power Engineering |
| `02_communications.py` | Chapter 2 — Communications Engineering |
| `03_semiconductors.py` | Chapter 3 — Semiconductors |
| `04_control_systems.py` | Chapter 4 — Control Systems |
| `05_embedded_systems.py` | Chapter 5 — Embedded Systems |
| `06_digital_logic.py` | Chapter 6 — Digital Logic |
| `07_circuit_analysis.py` | Chapter 7 — Circuit Analysis |
| `08_signal_processing.py` | Chapter 8 — Signal Processing |
| `09_electromagnetics.py` | Chapter 9 — Electromagnetics |
| `10_power_electronics.py` | Chapter 10 — Power Electronics |
| `11_instrumentation.py` | Chapter 11 — Instrumentation and Measurement |
| `12_electric_motors.py` | Chapter 12 — Electric Motors |
| `13_op_amps.py` | Chapter 13 — Operational Amplifiers |
| `14_appendix_a_phasors.py` | Appendix A — Imaginary Numbers and Phasors |
| `15_appendix_b_atan_vs_atan2.py` | Appendix B — Arctangent and atan2 |
| `15_networking.py` | Chapter 15 — Networking |
| `16_antenna_design.py` | Chapter 16 — Antenna Design and Principles |
| `16_appendix_c_decibels.py` | Appendix C — Decibels |
| `17_radar_systems.py` | Chapter 17 — Radar Systems |
| `18_optics.py` | Chapter 18 — Optics |
| `19_engineering_economics.py` | Chapter 19 — Engineering Economics |

## Requirements

- Python 3.9+
- marimo, numpy, matplotlib

```bash
pip install -r requirements.txt
```

## Running a Script

```bash
marimo run 04_control_systems.py
```

This opens the notebook in your browser. Each notebook contains interactive graphs and visualizations for the corresponding chapter.

To edit a notebook:

```bash
marimo edit 04_control_systems.py
```
