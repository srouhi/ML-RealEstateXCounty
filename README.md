# ML-RealEstateXCounty  

Real Estate Price Prediction for Small County Assessors

## Project Overview

This project explores methods to accurately predict housing market prices in smaller counties using machine learning models trained on both large outsourced datasets and limited local data. The goal is to determine whether a model trained solely on a small amount of local data or a model trained primarily on large external data supplemented by some local data yields more accurate predictions for local real estate pricing.

## Motivation

Small county assessors often lack sufficient local data to build reliable predictive models for property valuation. This project investigates if limited but relevant local data can outperform larger, less-relevant datasets when used for price prediction.

## Data Description

* **Primary Outsourced Dataset (King County):**

  * Over 21,000 entries with 21 housing features
  * Features include square footage, number of bedrooms, floors, grade, condition, basement size, etc.
  * Data cleaned to remove outliers and irrelevant features.

* **Local Dataset (Lancaster County):**

  * Originally over 74,000 entries, filtered down to \~4,000 valid sales from 2016.
  * Only single-family homes considered to maintain consistency.
  * Features standardized and aligned with the outsourced dataset.

## Feature Engineering & Analysis

* Correlation analysis showed **square footage** and **grade** are the most significant predictors of price.
* Price normalization was applied using median and standard deviation to reduce outlier impact.
* Features were min-max scaled for uniformity.

## Models

Two feedforward neural network models were developed:

* **Model 1:** Trained on 100% of King County data plus a small percentage of Lancaster data.
* **Model 2:** Trained solely on a small percentage of Lancaster data.

### Model Architecture

* Input layer with 64 neurons (one per feature), ReLU activation
* Three hidden layers: 64, 32, 16 neurons, ReLU activation
* Output layer: predicts property price, optimized with Mean Squared Error (MSE)

## Experiments

* Tested various training data sizes from Lancaster County (30%, 20%, 10%) alongside full King County data for Model 1.
* Model 2 trained only on varying percentages of Lancaster data.
* Results consistently showed Model 2 (local data only) achieved lower MSE, indicating better prediction accuracy.
* A minimum threshold of \~20% local data is needed for reliable performance; below this, accuracy degrades significantly.

## Results Summary

| Case | Training Data Size (Lancaster) | Model 1 MSE (King+Lancaster) | Model 2 MSE (Lancaster only) |
| ---- | ------------------------------ | ---------------------------- | ---------------------------- |
| 1    | 30%                            | 4.45e9                       | 1.73e9                       |
| 2    | 20%                            | 4.66e9                       | 1.60e9                       |
| 3    | 10%                            | 7.31e9                       | 2.25e9                       |

*Lower MSE indicates better model performance.*

## Conclusion

Using limited but relevant local data leads to more accurate housing price predictions than relying on larger, unrelated datasets. Data relevance outweighs data volume in this context. However, training datasets smaller than 20% of available local data can reduce prediction quality. Future work can improve accuracy by including additional features like internal home amenities and local environmental factors.

### Prerequisites

* Python 3
* Libraries: TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib

## Acknowledgements

* Data sourced from King County and Lancaster County Assessor Offices.
* Inspired by real estate analytics research and small-county assessor needs.

