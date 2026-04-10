#!/usr/bin/env python
"""Test the model on the generated audio samples."""

import librosa
import numpy as np
import joblib
from utils import extract_features

# Load the audio samples
healthy, sr = librosa.load('data/audio_samples/healthy_voice.wav', sr=22050)
parkinsons, sr = librosa.load('data/audio_samples/parkinsons_voice.wav', sr=22050)

# Extract features
print('Extracting features...')
features_healthy = extract_features(healthy, sr)
features_parkinsons = extract_features(parkinsons, sr)

# Load model and scaler
model = joblib.load('models/parkinsons_svm_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Scale features
features_healthy_scaled = scaler.transform([features_healthy])
features_parkinsons_scaled = scaler.transform([features_parkinsons])

# Get predictions and scores
pred_healthy = model.predict(features_healthy_scaled)[0]
score_healthy = model.decision_function(features_healthy_scaled)[0]

pred_parkinsons = model.predict(features_parkinsons_scaled)[0]
score_parkinsons = model.decision_function(features_parkinsons_scaled)[0]

print(f'\nHealthy voice:')
print(f'  Decision score: {score_healthy:.4f}')
print(f'  Threshold: {model.threshold:.4f}')
print(f'  Prediction: {"Parkinsons" if pred_healthy == 1 else "Healthy"}')

print(f'\nParkinson\'s voice:')
print(f'  Decision score: {score_parkinsons:.4f}')
print(f'  Threshold: {model.threshold:.4f}')
print(f'  Prediction: {"Parkinsons" if pred_parkinsons == 1 else "Healthy"}')
