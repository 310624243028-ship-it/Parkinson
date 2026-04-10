#!/usr/bin/env python
"""Check if Recording.wav can be tested."""

import librosa
import joblib
import numpy as np
import sys
sys.path.insert(0, '.')
from utils import extract_features

# Load the recording
print("Loading Recording.wav...")
try:
    y, sr = librosa.load('data/audio_samples/Recording.wav', sr=22050)
    print(f"✓ Loaded: {len(y)} samples at {sr}Hz\n")
    
    # Extract features
    print("Extracting features...")
    features = extract_features(y, sr)
    print(f"✓ Features extracted\n")
    
    # Load model and scaler
    model = joblib.load('models/parkinsons_svm_model.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    score = model.decision_function(features_scaled)[0]
    pred = model.predict(features_scaled)[0]
    
    print("=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"\nModel threshold: {model.threshold:.4f}")
    print(f"Your voice score: {score:.4f}")
    pred_text = "Parkinsons" if pred == 1 else "Healthy"
    print(f"Prediction: {pred_text}")
    print(f"Distance from threshold: {score - model.threshold:.4f}")
    
    if score < model.threshold:
        print("\n✓ Within healthy range (negative)")
    else:
        print("\n⚠️  Above threshold - classified as Parkinson's")
        print(f"   Need to increase threshold by at least {score - model.threshold:.2f}")
    
    print("\n" + "=" * 60)
    print("TOP 5 FEATURES BY VALUE")
    print("=" * 60)
    
    feature_names = [
        "Zero-crossing rate", "Spectral centroid", "Spectral rolloff", "Spectral bandwidth",
        "RMS mean", "RMS std", "Jitter", "Shimmer", "HNR",
        "MFCC 1", "MFCC 2", "MFCC 3", "MFCC 4", "MFCC 5", "MFCC 6", "MFCC 7",
        "Chroma 1", "Chroma 2", "Chroma 3", "Chroma 4", "Chroma 5", "Chroma 6"
    ]
    
    feature_values = list(zip(feature_names, features))
    feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (name, val) in enumerate(feature_values[:5], 1):
        print(f"{i}. {name}: {val:.6f}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
