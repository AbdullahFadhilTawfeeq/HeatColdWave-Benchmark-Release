[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18601489.svg)](https://doi.org/10.5281/zenodo.18601489)

# Heatwave and Coldwave Prediction Benchmark under Severe Class Imbalance

This repository provides a **public, reproducible implementation** of a comprehensive benchmarking framework for
heatwave and coldwave prediction using deep learning models applied to daily meteorological observations from
ground-based weather stations.

The framework is designed to systematically evaluate **temporal, spatial, spatio-temporal, and attention-based**
neural network architectures under **severe class imbalance** and **multi-horizon forecasting** settings.

---

## Scope of the Repository

This repository **contains only the code required to reproduce the modeling framework** described in the associated manuscript.

### Included
- Core Python implementation of the benchmark framework
- Model training, validation, and evaluation pipelines
- Cost-sensitive learning for extreme-event imbalance
- Multi-horizon forecasting (e.g., 1–7 days ahead)
- Day-level and spell-level (event-based) evaluation procedures
- Bayesian hyperparameter optimization (Optuna)

### Not Included
- Meteorological datasets
- Trained model weights
- Optuna databases or experimental results

These materials are intentionally excluded to comply with data-sharing restrictions and to keep the repository lightweight.
---

## Evaluation Strategy

Model performance is assessed using complementary evaluation perspectives:
- **Multi-horizon forecasting** (e.g., 1, 3, 5, and 7 days ahead)
- **Day-level evaluation** for fine-grained classification performance
- **Spell-level (event-based) evaluation** to assess event persistence and operational relevance

Spell-based evaluation treats each extreme event as a continuous episode rather than isolated daily predictions,
providing a more realistic assessment for climate-risk applications.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/HeatColdWave-Benchmark-Release.git
cd HeatColdWave-Benchmark-Release
