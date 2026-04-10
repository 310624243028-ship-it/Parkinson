#!/usr/bin/env python
"""Test synthetic audio samples - simpler version without extract_features."""

import numpy as np
import librosa
import joblib
import pandas as pd

print("Loading audio and model...")

# Load synthetic audio
healthy_audio, sr = librosa.load('data/audio_samples/healthy_voice.wav', sr=22050)
parkinsons_audio, sr = librosa.load('data/audio_samples/parkinsons_voice.wav', sr=22050)

print(f"Healthy audio shape: {healthy_audio.shape}, sr: {sr}")
print(f"Parkinsons audio shape: {parkinsons_audio.shape}, sr: {sr}")

# Load model and scaler
model = joblib.load('models/parkinsons_svm_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# For now, let's just extract basic features manually
print("\nExtracting basic features...")

# Healthy features
zcr_h = np.mean(librosa.feature.zero_crossing_rate(healthy_audio))
spec_cent_h = np.mean(librosa.feature.spectral_centroid(y=healthy_audio, sr=sr))
print(f"\nHealthy voice basic features:")
print(f"  Zero-crossing rate: {zcr_h:.6f}")
print(f"  Spectral centroid: {spec_cent_h:.6f}")

# Parkinsons features  
zcr_p = np.mean(librosa.feature.zero_crossing_rate(parkinsons_audio))
spec_cent_p = np.mean(librosa.feature.spectral_centroid(y=parkinsons_audio, sr=sr))
print(f"\nParkinson's voice basic features:")
print(f"  Zero-crossing rate: {zcr_p:.6f}")
print(f"  Spectral centroid: {spec_cent_p:.6f}")

print(f"\nModel threshold: {model.threshold:.4f}")
print("\nNote: The issue is that synthetic audio features don't match real speech.")
print("Real speech has different acoustic characteristics than generated sine waves.")
