# Corrosion Rate Prediction - Fuel System Materials

Predicting the corrosion rate of metal components in fuel systems using environmental, chemical, and material process parameters. Built with scikit-learn and XGBoost, comparing linear, bagging, and boosting regression models.

## Problem Statement

Metal parts in fuel systems (pipes, tanks, fittings) corrode over time due to exposure to ethanol, water, chlorides, dissolved oxygen, and other factors. This project builds a regression pipeline that predicts `corrosion_rate_mm_per_y` (millimeters per year) from 13 measurable operating conditions, so corrosion risk can be estimated without waiting years for real-world exposure data.

## Dataset

- **Rows:** 10,000
- **Columns:** 14 (13 features + 1 target)
- **Target:** `corrosion_rate_mm_per_y`
- **No missing values**

| Feature | Description |
|---|---|
| `material` | Metal type (Brass, Al6061, SS304, LowCarbonSteel, Copper, ZnCoatedSteel) |
| `ethanol_pct` | Ethanol percentage in fuel |
| `temperature_C` | Operating temperature (Celsius) |
| `water_ppm` | Water content (parts per million) |
| `chloride_mg_L` | Chloride concentration (mg/L) |
| `dissolved_O2_ppm` | Dissolved oxygen (ppm) |
| `TAN_mgKOH_g` | Total Acid Number |
| `conductivity_uS_cm` | Electrical conductivity |
| `flow_velocity_m_s` | Fluid flow velocity |
| `exposure_hours` | Duration of exposure |
| `surface_roughness_Ra_um` | Surface roughness |
| `inhibitor_present` | Whether a corrosion inhibitor was used (0/1) |
| `pH` | pH of the fluid |

## Project Workflow

1. **Exploratory Data Analysis**
   Checked data types, missing values, target distribution, material distribution, and feature correlations via a heatmap.

2. **Preprocessing**
   - `material` (a nominal categorical feature) was One-Hot Encoded instead of Label Encoded, to avoid implying a false ordinal relationship between materials.
   - Data was split 80/20 into train and test sets with `random_state=42`.
   - `StandardScaler` was applied only where it matters: linear models (Linear Regression, Ridge, Lasso), since they are sensitive to feature magnitude. Tree based models (Random Forest, Gradient Boosting, AdaBoost, XGBoost) were trained on unscaled features.

3. **Model Training**
   Seven regression models were trained and evaluated on the same test set using R2, RMSE, and MAE.

4. **Model Comparison**
   Results were collected into a single comparison table and visualized with a bar chart.

5. **Feature Importance**
   Extracted and plotted feature importances from the tree based models (Random Forest, Gradient Boosting, XGBoost) to identify the strongest drivers of corrosion.

6. **Cross-Validation**
   Ran 5-fold cross-validation on the best performing model to confirm the result was not a lucky split.

## Models Used

| Model | Key Hyperparameters |
|---|---|
| Linear Regression | defaults |
| Ridge Regression | defaults (alpha=1.0) |
| Lasso Regression | defaults (alpha=1.0) |
| Random Forest Regressor | n_estimators=300, random_state=42 |
| Gradient Boosting Regressor | n_estimators=400, learning_rate=0.1, max_depth=3, random_state=42 |
| AdaBoost Regressor | defaults, random_state=42 |
| XGBoost Regressor | n_estimators=400, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42 |

## Results

| Model | R2 Score | RMSE | MAE |
|---|---|---|---|
| **XGBoost (400 Trees)** | **0.8884** | **0.006072** | 0.003371 |
| Random Forest (300 Trees) | 0.8844 | 0.006179 | 0.003327 |
| Gradient Boosting (400 Trees) | 0.8842 | 0.006185 | 0.003371 |
| Linear Regression | 0.7519 | 0.009052 | 0.005670 |
| Ridge Regression | 0.7519 | 0.009052 | 0.005670 |
| AdaBoost | 0.6854 | 0.010195 | 0.008611 |
| Lasso Regression | -0.0023 | 0.018195 | 0.012711 |

**Best model: XGBoost**, confirmed with 5-fold cross-validation:
- Fold R2 scores: `[0.8969, 0.9034, 0.8809, 0.9067, 0.8989]`
- Mean CV R2: **0.8974 (+/- 0.0089)**

## Key Findings

- One-Hot Encoding instead of Label Encoding lifted Linear Regression's R2 from 0.264 to 0.752, showing how much encoding choice matters for linear models on nominal categorical data.
- Linear models plateau around R2 of 0.75, since the true relationship between operating conditions and corrosion rate is meaningfully nonlinear.
- Lasso's default regularization strength (alpha=1.0) zeroed out too many coefficients on this feature set, producing a negative R2, worse than simply predicting the mean every time. This shows the importance of tuning regularization strength rather than relying on defaults.
- Tree based ensembles (Random Forest, Gradient Boosting, XGBoost) all land close together around R2 of 0.88, since they all capture the same nonlinear feature interactions. XGBoost edges ahead due to its regularization and optimized boosting procedure.
- 5-fold cross-validation confirms the XGBoost result generalizes and is not an artifact of one lucky train/test split.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- XGBoost
- matplotlib, seaborn

## Project Structure

```
.
├── corrosion_fuel.csv
├── Ecorr_file_enhanced.ipynb
└── README.md
```

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
2. Make sure `corrosion_fuel.csv` is in the same directory as the notebook.
3. Open and run `Ecorr_file_enhanced.ipynb` top to bottom in Jupyter Notebook, JupyterLab, or VS Code.

## Possible Next Steps

- Hyperparameter tuning via GridSearchCV or RandomizedSearchCV for XGBoost and Random Forest
- SHAP values for deeper, per-prediction feature importance
- Try additional models such as LightGBM or a stacked ensemble
- Deploy the best model behind a simple API for real time corrosion rate estimation
