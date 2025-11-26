# Corrosion Rate Prediction using Machine Learning
### By Anjaneya Parashar

## Project Overview
This project focuses on predicting the corrosion rate (mm/year) of various metal samples based on fuel type and environmental parameters. Using multiple machine learning models — from simple linear models to advanced boosting techniques — the goal is to identify the most accurate and reliable method for corrosion prediction.

The dataset consists of:
- Material type
- Sample dimensions and mass
- Exposure duration
- Corrosion losses
- Final computed corrosion rate

## Dataset Description
Dataset Name: corrosion_fuel.csv

Contains the following key columns:
- material: Type of material tested
- init_mass: Initial mass of sample
- final_mass: Final mass after corrosion test
- length / width / thickness: Sample dimensions
- area: Exposed area
- exposure_time: Hours of exposure
- corrosion_loss: Mass loss
- corrosion_rate_mm_per_y: Target variable

## Machine Learning Models Used
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Gradient Boosting Regressor (400-tree ensemble)
5. AdaBoost Regressor

## Train-Test Split
80–20 split:
- 80% training
- 20% testing

## Evaluation Metrics
Linear, Ridge, Lasso:
- R² Score
- RMSE

Gradient Boosting & AdaBoost:
- R² Score
- RMSE
- MAE

## Key Results
- Linear, Ridge, and Lasso provide baseline performance.
- Gradient Boosting (400 trees) achieves best R² and lowest RMSE/MAE.
- AdaBoost performs slightly worse than Gradient Boosting.

## Final Conclusion
Gradient Boosting (400-tree ensemble) is the most effective model for predicting corrosion rate. It captures nonlinear behavior and provides minimal error values.

## Project Structure
corrosion_fuel.csv  
Corrosion_Prediction.ipynb  
README.md

## Author
Anjaneya Parashar
