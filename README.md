# Feature Engineering Capstone
## StaySmart Hotels - Cancellation Risk Prediction

This project predicts hotel booking cancellations using the Hotel Bookings dataset.
The goal is to show that feature engineering and preprocessing improve model performance.

## Dataset
Hotel Bookings dataset - 119,380 rows, 33 columns
Target: is_canceled (binary classification)

## Results Summary
| Version | ROC-AUC | F1 |
|---------|---------|-----|
| Baseline | 0.9051 | 0.7257 |
| After Preprocessing | 0.9558 | 0.8447 |
| After Feature Engineering | 0.9809 | 0.9066 |
| After Selection (Top 20) | 0.9739 | 0.8855 |

## How to Run
1. Open FeatureEngineering_Capstone.ipynb in Google Colab
2. Runtime > Restart and Run All
3. No local setup needed, dataset loads from URL automatically

## Requirements
See requirements.txt

## Repo Structure
- FeatureEngineering_Capstone.ipynb - main notebook
- src/helpers.py - reusable feature construction functions
- report/ - contains Report PDF
- requirements.txt - dependencies
