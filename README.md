# K-Nearest Neighbours for Binary Classification

This project implements a **K-Nearest Neighbours (KNN)** algorithm for **binary classification** from scratch using Python. The classifier predicts labels based on majority voting among nearest neighbors and evaluates performance using the **F1-score**.

## Project Overview
- Binary classification using KNN
- No external machine learning libraries used
- Performance evaluated with F1-score
- Hyperparameters tuned to select the best model

## Features
- Distance metrics:
  - Euclidean distance
  - Minkowski distance (p = 3)
  - Cosine distance
- Data preprocessing:
  - Feature normalization
  - Min-Max scaling
- Hyperparameter tuning:
  - Number of neighbors (k)
  - Distance function
  - Scaling method

## Implementation
- `knn.py`: KNN training, neighbor selection, and prediction
- `utils.py`: Distance functions, F1-score, data scalers, and hyperparameter tuning

## How to Run
```bash
python test.py

