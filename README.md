# Retail Demand Forecasting Solution (Walmart Dataset)

This repository contains an end-to-end machine learning pipeline for forecasting retail demand, using the Kaggle [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) dataset.

## Project Overview

The goal is to predict department-wide weekly sales for 45 Walmart stores, incorporating external factors such as holidays, markdowns, temperature, fuel prices, and CPI.

Three models were built and compared:
1. **Linear Regression** — simple baseline
2. **Random Forest Regressor** — ensemble of decision trees
3. **XGBoost (Optuna-tuned)** — gradient-boosted trees with Bayesian hyperparameter optimization

### Key Results

| Model | MAE | RMSE | WMAE | MAPE | R² |
|---|---|---|---|---|---|
| Linear Regression | 1,736.57 | 3,294.34 | 1,799.89 | 136.68% | 0.977 |
| Random Forest | 1,297.85 | 2,748.94 | 1,381.83 | **91.63%** | 0.984 |
| **XGBoost (Tuned)** | **1,203.07** | **2,494.95** | **1,283.31** | 166.44% | **0.987** |

**Best Model:** XGBoost (Tuned) — achieves the lowest WMAE (the official Kaggle competition metric, which penalizes holiday-week errors 5×).

#### A Note on Model Selection (WMAE vs. MAPE)
I selected **XGBoost** based on the official evaluation metric used for the Walmart dataset: **Weighted Mean Absolute Error (WMAE)**. WMAE assigns a 5× higher weight to errors made during holiday weeks, as forecasting demand accurately during holidays is critical for retail inventory planning.

Although Random Forest achieved a slightly better MAPE, MAPE measures *percentage error* and can become disproportionately large when actual sales are close to zero (a common occurrence in this dataset). In contrast, WMAE evaluates the model based on weighted absolute dollar errors, which is the exact metric specified by Walmart for this competition. Furthermore, XGBoost achieved the **lowest WMAE (1283.31), lowest MAE (1203.07), and highest R² (0.987)** among all models, making it the definitive optimal choice.

## Dataset

From the Kaggle competition: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)

| File | Description |
|------|-------------|
| `stores.csv` | Store metadata — type (A, B, C) and size (sq ft) |
| `train.csv` | Historical weekly sales: Store, Dept, Date, Weekly_Sales, IsHoliday |
| `test.csv` | Future weeks for prediction (no Weekly_Sales) |
| `features.csv` | External features: Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment |

## Folder Structure
```
 demand-forecasting/
 ├── data/
 │   └── raw/                    # Downloaded Kaggle CSVs (included)
 ├── notebooks/
 │   └── demand_forecasting.ipynb  # Main deliverable — runs end-to-end
 ├── reports/
 │   ├── Process_Flow_Document.pdf          # Process flow diagrams + detailed explanations
 │   └── Model_Comparison_Evaluation_Report.pdf # Model selection reasoning and trade-offs
 ├── src/
 │   └── metrics.py                # Custom evaluation metrics
 ├── .gitignore
 ├── README.md
 └── requirements.txt
```

## Setup & Reproduction Instructions

### 1. Environment Setup
```bash
conda create -n ml python=3.10 -y
conda activate ml
pip install -r requirements.txt
```

### 2. Data Download
Download the dataset from Kaggle (you must accept competition rules first):
```bash
kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p data/raw
# Extract the CSVs into data/raw/
```

### 3. Run the Notebook
```bash
jupyter notebook notebooks/demand_forecasting.ipynb
```
Select **Kernel → Restart & Run All** to execute the entire pipeline end-to-end.

Alternatively, run headlessly:
```bash
jupyter nbconvert --to notebook --execute notebooks/demand_forecasting.ipynb --output demand_forecasting.ipynb
```

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Jupyter Notebook (.ipynb) | `notebooks/demand_forecasting.ipynb` |
| 2 | Process Flow Document (PDF) | `reports/Process_Flow_Document.pdf` |
| 3 | Model Comparison Report (PDF)| `reports/Model_Comparison_Evaluation_Report.pdf` |
| 4 | README.md | This file |
| 5 | requirements.txt | `requirements.txt` |

## References
- [Walmart Recruiting — Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
