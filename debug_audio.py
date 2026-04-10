#!/usr/bin/env python
"""Debug script to test audio processing step by step."""

import librosa
import numpy as np
import sys
sys.path.insert(0, '.')
from utils import extract_features

# Test with the synthetic audio
print("Testing audio processing...\n")

try:
    print("1. Loading audio...")
    y, sr = librosa.load('data/audio_samples/healthy_voice.wav', sr=22050)
    print(f"   ✓ Loaded: {len(y)} samples at {sr}Hz\n")
    
    print("2. Extracting features...")
    print("   This may take a minute...")
    features = extract_features(y, sr)
    print(f"   ✓ Success! Extracted {len(features)} features\n")
    
    print("3. Features extracted:")
    for i, f in enumerate(features):
        print(f"   Feature {i}: {f:.6f}")
        
except Exception as e:
    print(f"   ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
