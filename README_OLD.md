# Cr³⁺ Dq/B Crystal Field Parameter Prediction

> Machine learning models for predicting the crystal field splitting parameter **Dq/B** in Cr³⁺ complexes using physicochemical and structural descriptors.

---

## Author

| | |
|---|---|
| **Name** | Snežana (Miladinović, Dragan) Đurković |
| **Affiliation** | INN "Vinča", Belgrade, Serbia — OMAS Group (Optical Materials and Spectroscopy Group), headed by Prof. Dr. Miroslav Dramićanin |
| **Email** | snezana.djurkovic@vin.bg.ac.rs |
| **Year** | 2026 |

---

## Description

This repository contains two machine learning models developed for predicting the crystal field splitting parameter Dq/B in octahedral Cr³⁺ complexes:

- A **white-box model** based on gradient boosting (CatBoost) — interpretable, with feature importance
- A **black-box model** based on a deep neural network (MLP) trained via backpropagation — higher complexity, no direct interpretability (design choice)

Both models follow an identical pipeline:
- 10-fold cross-validation with automated random state optimization
- Ensemble-based uncertainty estimation
- Parity plot generation
- Excel output with predictions and uncertainties

---

## Repository Structure

```
CrIII-DqB-prediction/
│
├── nn_backprop_model.py          # Black-box MLP neural network (backpropagation)
├── nn_black_box_model.py         # Black-box MLP neural network (base version)
├── catboost_white_box_model.py   # White-box CatBoost gradient boosting model
│
├── Cr3_dgb_training_set_pročišćen_proširen.xlsx   # Training dataset
├── To_predict.xlsx                                 # External prediction set
│
├── README.md
└── LICENSE
```

---

## Models

### White-Box Model — CatBoost (`catboost_white_box_model.py`)

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosted Decision Trees |
| Depth | 3 |
| Iterations | 250 |
| Learning rate | 0.08 |
| L2 regularization | 1.75 |
| Loss function | RMSE |

- Interpretable via feature importance scores
- Fast training (~1–3 min for full pipeline)

### Black-Box Model — Neural Network (`nn_backprop_model.py`)

| Parameter | Value |
|---|---|
| Algorithm | Multilayer Perceptron (MLP) |
| Hidden layers | 4 (128 → 64 → 32 → 16) |
| Epochs | 500 |
| Batch size | 64 |
| L2 weight decay | 1.9 × 10⁻³ |
| Optimizer | Adam + Cosine Annealing LR |
| Activation | ReLU + BatchNorm + Dropout |

- Trained via backpropagation with gradient clipping
- He (Kaiming) weight initialization

---

## Pipeline

Both models follow the same evaluation pipeline:

```
Load data
    │
    ▼
Search best random state (29 candidates × 10 folds = 290 model trainings)
    │
    ▼
Final 10-fold cross-validation (best random state)
    │
    ▼
Train final model on full dataset
    │
    ▼
Predict + estimate uncertainty (std of fold predictions)
    │
    ▼
Save Excel output + Parity plot
```

---

## Requirements

Install all dependencies with:

```bash
pip install torch pandas numpy scikit-learn matplotlib openpyxl catboost
```

| Package | Purpose |
|---|---|
| `torch` | Neural network (PyTorch) |
| `catboost` | Gradient boosting model |
| `pandas` | Data loading and Excel I/O |
| `numpy` | Numerical computations |
| `scikit-learn` | Preprocessing, CV, metrics |
| `matplotlib` | Parity plot visualization |
| `openpyxl` | Excel file writing |

---

## Usage

1. Place your Excel files in the same folder as the scripts
2. Open Command Prompt and navigate to that folder:

```bash
cd C:\Users\Korisnik\Downloads
```

3. Run the desired model:

```bash
# Neural network (black-box)
python nn_backprop_model.py

# CatBoost (white-box)
python catboost_white_box_model.py
```

4. Output files will be saved in the same folder:
   - `final_prediction_backprop.xlsx` — predictions + uncertainty
   - `parity_plot_backprop.png` — parity plot

---

## Output

Each model produces:

| File | Content |
|---|---|
| `final_prediction_*.xlsx` | Formula, Predicted Dq/B, Uncertainty |
| `parity_plot_*.png` | True vs Predicted scatter plot |
| Console | R² per fold, final R², MAE |

---

## Citation

If you use this code in your research, please cite:

```
Snežana Đurković (2026). Machine learning prediction of Dq/B crystal field
splitting parameter in Cr³⁺ complexes. GitHub repository:
https://github.com/KirkaSSS/phd
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
