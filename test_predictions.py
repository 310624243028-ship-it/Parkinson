import librosa
import numpy as np
import joblib
from pathlib import Path
import os
from utils import extract_features

# Change to script directory
os.chdir(Path(__file__).parent)

# Load model and scaler
model = joblib.load('models/parkinsons_svm_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Load test audio
healthy_path = 'data/audio_samples/healthy_voice.wav'
parkinsons_path = 'data/audio_samples/parkinsons_voice.wav'

if Path(healthy_path).exists() and Path(parkinsons_path).exists():
    healthy, sr = librosa.load(healthy_path, sr=22050)
    parkinsons_audio, sr = librosa.load(parkinsons_path, sr=22050)
    
    # Extract features
    features_h = extract_features(healthy, sr)
    features_p = extract_features(parkinsons_audio, sr)
    
    # Scale features
    features_h_scaled = scaler.transform(features_h.reshape(1, -1))
    features_p_scaled = scaler.transform(features_p.reshape(1, -1))
    
    # Get predictions
    pred_h = model.predict(features_h_scaled)
    pred_p = model.predict(features_p_scaled)
    
    score_h = model.decision_function(features_h_scaled)
    score_p = model.decision_function(features_p_scaled)
    
    print("HEALTHY VOICE:")
    print(f"  Prediction: {'Parkinsons' if pred_h[0] == 1 else 'Healthy'}")
    print(f"  Decision Score: {score_h[0]:.4f}")
    print(f"  First 5 extracted features: {features_h[:5]}")
    print(f"  First 5 scaled features: {features_h_scaled[0][:5]}")
    
    print("\nPARKINSONS VOICE:")
    print(f"  Prediction: {'Parkinsons' if pred_p[0] == 1 else 'Healthy'}")
    print(f"  Decision Score: {score_p[0]:.4f}")
    print(f"  First 5 extracted features: {features_p[:5]}")
    print(f"  First 5 scaled features: {features_p_scaled[0][:5]}")
else:
    print("Audio files not found!")
    print(f"  Healthy path exists: {Path(healthy_path).exists()}")
    print(f"  Parkinsons path exists: {Path(parkinsons_path).exists()}")
