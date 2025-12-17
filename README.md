K-Nearest Neighbours for Binary Classification

This project implements a K-Nearest Neighbours (KNN) classifier for binary classification from scratch using Python. The model predicts labels using majority voting among the nearest neighbors and evaluates performance using the F1-score.

What This Project Does

Implements KNN without using external ML libraries

Supports multiple distance metrics:

Euclidean distance

Minkowski distance (p = 3)

Cosine distance

Computes F1-score for model evaluation

Applies data preprocessing techniques:

Feature normalization

Min-Max scaling

Performs hyperparameter tuning to select the best:

Number of neighbors (k)

Distance function

Scaling method

Implementation Details

knn.py contains the KNN model logic (training, neighbor selection, prediction)

utils.py contains:

Distance functions

F1-score computation

Data scalers

Hyperparameter tuning logic

How to Run
python test.py

Output

Prints F1-scores for different configurations

Selects and evaluates the best-performing KNN model

Notes

Designed for binary classification (labels 0 and 1)

No external libraries used beyond the provided starter code
