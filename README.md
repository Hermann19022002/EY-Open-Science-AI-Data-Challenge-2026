# EY Open Science AI & Data Challenge 2026 — Water Quality Prediction

> **3rd place — France finals** · Team **Data4Decision** (Hermann Banzouzi Miampassi & Neville Tchatchou Njatcha)
>
> Predicting three water quality indicators across South African rivers using satellite and climate data, with a focus on **spatial generalization** to regions never seen during training.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents

- [Context](#context)
- [The Data](#the-data)
- [Approach](#approach)
- [Results](#results)
- [Key Insights](#key-insights)
- [Repository Structure](#repository-structure)
- [Installation & Reproduction](#installation--reproduction)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## Context

Access to safe drinking water remains a challenge for **2.1 billion people** worldwide. In South Africa, around **70% of large river systems are eutrophic or hypereutrophic**, and field measurements remain slow and expensive.

The EY Open Science AI & Data Challenge 2026 asked us to predict three water quality parameters on South African rivers from satellite and climate data:

| Parameter | Name | Unit | Meaning |
|-----------|------|------|---------|
| **TA** | Total Alkalinity | mg/L | Buffering capacity against acidification |
| **EC** | Electrical Conductance | µS/cm | Proxy for mineralization / salinity |
| **DRP** | Dissolved Reactive Phosphorus | mg/L | Key eutrophication indicator |

**Evaluation metric**: mean R² across the three targets, on **24 validation sites not seen during training**.

---

## The Data

| Dataset | Size | Source |
|---------|------|--------|
| Training | 9,319 observations · 162 stations · 2011–2015 | EY Challenge |
| Validation | 200 points · 24 unseen stations | EY Challenge |
| Landsat 7/8 | Spectral reflectances (NIR, Green, SWIR) | Google Earth Archive |
| TerraClimate | PET, precipitation, temperature, runoff | UCAR / Climatology Lab |

**The core difficulty**: the 24 validation stations are **80–280 km** away from their nearest training neighbor (median: 190 km). Random cross-validation splits massively overestimate generalization performance on this kind of spatial transfer problem.

---

## Approach

### Feature engineering

After systematic leaderboard probing with feature sets ranging from 13 to 65 variables, the best generalization was obtained with **13 carefully selected features**:

| Category | Features |
|----------|----------|
| **Landsat** | `nir`, `green`, `swir16`, `swir22`, `NDMI`, `MNDWI` |
| **Climate** | `pet` (potential evapotranspiration) |
| **Temporal** | `Month`, `Season`, `Month_sin`, `Month_cos` |
| **Geographic** | `Latitude`, `Longitude` |

### Model selection — a hybrid strategy

Each target has a different statistical nature; no single algorithm is optimal for all three.

| Target | Model | Rationale |
|--------|-------|-----------|
| TA | **Random Forest Extreme** (500 trees, `max_features="sqrt"`) | Robust to outliers, captures non-linear geographic patterns |
| EC | **Random Forest Extreme** | Correlated with mineralization; RF is stable across geographic shifts |
| DRP | **LightGBM** (selected against RF & XGBoost) | Highly skewed distribution (γ₁ = 1.64); gradient boosting handles long tails better |

### Honest validation

We used **GroupKFold by station** (5 folds, groups = stations) rather than random splits, which is the only unbiased estimator of the spatial transfer we are measured on.

---

## Results

### Internal validation (80/20 split)

| Target | Model | R² | MAE |
|--------|-------|-----|-----|
| TA | RF Extreme | 0.7883 | 23.82 |
| EC | RF Extreme | 0.8211 | 94.64 |
| DRP | LightGBM | 0.6964 | 15.95 |
| **Mean** | | **0.7686** | |

---

## Key Insights

1. **Simplicity generalizes better.** Going from 65 features to 13 **doubled** our leaderboard score. Buffer statistics and land-cover features looked very informative on random splits but caused severe spatial overfitting.
2. **Validation must match the problem structure.** A random 80/20 split overestimated performance by +0.41 R² compared to leaderboard reality.
3. **Tree ensembles beat neural networks on small tabular data with spatial shift.** Our 3-branch neural network scored **R² = −0.41** on spatial folds — catastrophically worse than RF + LightGBM.
4. **Data enrichment should be guided by physics**, not by training-set performance. Variables that aren't physically meaningful to water chemistry don't transfer geographically.

---

## Repository Structure

```
.
├── solution_finale.py       # Single reproducible pipeline (training + prediction)
├── submission_finale.csv    # Final submission file (LB = 0.3629)
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

---

## Installation & Reproduction

### Requirements
- Python 3.10+
- Around 4 GB of RAM

### Setup

```bash
git clone https://github.com/Hermann19022002/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce the submission

```bash
python solution_finale.py
# → writes submission_finale.csv
```

---

## Team

| Member | Role | Links |
|--------|------|-------|
| **Hermann Banzouzi Miampassi** | Data Science · ML modeling | [GitHub](https://github.com/Hermann19022002) |
| **Neville Tchatchou Njatcha** | Data Science · Feature engineering | — |

Both are Data Scientist students at **ENSAI** (École Nationale de la Statistique et de l'Analyse de l'Information), Bruz, France.

---

## Acknowledgments

- **EY** and the Open Science AI & Data Challenge team for organizing the competition
- **ENSAI** for relaying the challenge invitation
- **NASA / USGS** for Landsat data
- **Climatology Lab (UCSB)** for TerraClimate
- **Dr Emma Haziza** for her inspiring keynote on water resilience

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Abatzoglou, J.T. et al. (2018). *TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance.* Scientific Data.
- Gorelick, N. et al. (2017). *Google Earth Engine: Planetary-scale geospatial analysis for everyone.* Remote Sensing of Environment.
- Roberts, D.R. et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography.

---

<p align="center">
  <i>"We didn't build just a model — we built a decision tool for those who manage water."</i>
</p>
