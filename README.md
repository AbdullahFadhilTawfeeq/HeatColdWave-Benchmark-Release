# Heatwave and Coldwave Prediction Benchmark under Severe Class Imbalance

This repository provides the implementation of a comprehensive benchmarking framework for heatwave and coldwave detection using deep learning models applied to daily meteorological observations from ground-based weather stations.

The framework is designed to systematically evaluate temporal, spatial, spatio-temporal, and attention-based neural network architectures under severe class imbalance and multi-horizon forecasting settings.

---

##  Scope of the Repository

This repository **contains only the code required to reproduce the modeling framework** described in the associated manuscript.

### Included
- Core Python implementation of the benchmark framework
- Model training, evaluation, and comparison logic
- Cost-sensitive learning implementation
- Multi-horizon forecasting pipeline
- Day-level and spell-level evaluation procedures

---

## Handling Class Imbalance

Extreme heatwave and coldwave events are rare, leading to severe class imbalance.

To address this challenge, the framework employs:
- Optimized **cost-sensitive learning** via weighted cross-entropy
- **Macro F1-score–driven** hyperparameter optimization and early stopping
- A dedicated **ablation analysis** comparing cost-sensitive learning with data-level resampling strategies

---

## Evaluation Strategy

Model performance is assessed using:
- **Multi-horizon forecasting** (1, 3, 5, and 7 days ahead)
- **Day-level evaluation** for fine-grained accuracy
- **Spell-level evaluation** to assess event persistence and operational relevance

Spell-based evaluation treats each extreme event as a continuous episode rather than isolated daily predictions.

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/HeatColdWave-Benchmark-Release.git
   cd HeatColdWave-Benchmark-Release
