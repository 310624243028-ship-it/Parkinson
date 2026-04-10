#!/usr/bin/env python
"""Check model threshold and statistics."""

import pandas as pd
import numpy as np
import joblib

# Load data
df = pd.read_csv('data/parkinsons.csv')
X = df.drop(['status', 'name'], axis=1)
y = df['status']

# Load model and scaler
model = joblib.load('models/parkinsons_svm_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Scale features
X_scaled = scaler.transform(X)

# Get decision scores for all training data
scores = X_scaled @ model.weights

# Get scores by class
scores_healthy = scores[y == 0]
scores_parkinsons = scores[y == 1]

print("=" * 60)
print("Model Threshold Analysis")
print("=" * 60)

print(f"\nModel threshold: {model.threshold:.4f}")

print(f"\nHealthy voice scores (n={len(scores_healthy)}):")
print(f"  Min: {scores_healthy.min():.4f}")
print(f"  Max: {scores_healthy.max():.4f}")
print(f"  Mean: {scores_healthy.mean():.4f}")
print(f"  Std: {scores_healthy.std():.4f}")

print(f"\nParkinson's scores (n={len(scores_parkinsons)}):")
print(f"  Min: {scores_parkinsons.min():.4f}")
print(f"  Max: {scores_parkinsons.max():.4f}")
print(f"  Mean: {scores_parkinsons.mean():.4f}")
print(f"  Std: {scores_parkinsons.std():.4f}")

# Check accuracy at current threshold
pred = (scores > model.threshold).astype(int)
accuracy = (pred == y).mean()
print(f"\nAccuracy at current threshold: {accuracy:.4f}")

# Count misclassifications
false_positives = ((pred == 1) & (y == 0)).sum()
false_negatives = ((pred == 0) & (y == 1)).sum()
print(f"False positives (healthy -> Parkinson's): {false_positives}")
print(f"False negatives (Parkinson's -> healthy): {false_negatives}")
