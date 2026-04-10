#!/usr/bin/env python
"""
Simple script to organize and test real audio samples.
"""

import os
from pathlib import Path
import librosa
import numpy as np

print("=" * 70)
print("Real Audio Sample Setup Guide")
print("=" * 70)

audio_dir = Path('data/audio_samples')
audio_dir.mkdir(parents=True, exist_ok=True)

print("\n📁 Audio Sample Directory: data/audio_samples/")
print("\n" + "=" * 70)
print("OPTION B: RECORD WITH WINDOWS SOUND RECORDER")
print("=" * 70)

print("""
Step 1: Open Sound Recorder
   - Press Windows key and R together
   - Type: soundrecorder
   - Or search for "Voice Memos" in Start menu

Step 2: Record HEALTHY voice sample
   - Click "New recording"
   - Say "aaaa" or "oooo" in a steady, clear voice for 3-5 seconds
   - Click Stop
   - Click "Save as" and name it "healthy_voice.wav"
   - Location: c:\\Users\\Brindaa\\OneDrive\\Desktop\\Parkinsons\\data\\audio_samples\\

Step 3: Record PARKINSONS-like voice sample
   - Start a new recording
   - Record another "aaaa" or "oooo" sample
   - Save as "parkinsons_voice.wav"
   - Same location as above

Expected Files After Recording:
* data/audio_samples/healthy_voice.wav
* data/audio_samples/parkinsons_voice.wav
""")

print("\n" + "=" * 70)
print("OPTION C: DOWNLOAD REAL PARKINSON'S DATASETS")
print("=" * 70)

print("""
Step 1: Download Datasets
   Run this command to see available sources:
   python scripts/download_datasets.py

Step 2: Choose a dataset and download
   Recommended: UCI ML Repository or Kaggle Parkinson's datasets
   
   Links:
   - UCI ML: https://archive.ics.uci.edu/ml/datasets/Parkinson's+Disease+Classification
   - Kaggle: https://www.kaggle.com/datasets/nikhilbhokare/parkinsons-voice-data
   - PVI: https://www.parkinsonsvoiceinitiative.org/

Step 3: Extract voice samples
   - Find .wav files from healthy subjects
   - Save as healthy_voice.wav
   - Find .wav files from Parkinson's patients
   - Save as parkinsons_voice.wav
   - Place in: data/audio_samples/

Step 4: Format Requirements
   * Format: WAV (or mp3, m4a, ogg)
   * Sample rate: any (auto-converted to 22050 Hz)
   * Duration: 3-5 seconds recommended
   * Content: Sustained vowel sounds work best
""")

print("\n" + "=" * 70)
print("VERIFY YOUR AUDIO FILES")
print("=" * 70)

# Check existing files
healthy_exists = (audio_dir / "healthy_voice.wav").exists()
parkinsons_exists = (audio_dir / "parkinsons_voice.wav").exists()

if healthy_exists:
    try:
        y, sr = librosa.load(audio_dir / "healthy_voice.wav", sr=22050)
        duration = len(y) / sr
        print(f"✓ healthy_voice.wav found")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Sample rate: {sr} Hz")
    except Exception as e:
        print(f"✗ Error reading healthy_voice.wav: {e}")
else:
    print("✗ healthy_voice.wav NOT found")

if parkinsons_exists:
    try:
        y, sr = librosa.load(audio_dir / "parkinsons_voice.wav", sr=22050)
        duration = len(y) / sr
        print(f"✓ parkinsons_voice.wav found")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Sample rate: {sr} Hz")
    except Exception as e:
        print(f"✗ Error reading parkinsons_voice.wav: {e}")
else:
    print("✗ parkinsons_voice.wav NOT found")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

if healthy_exists and parkinsons_exists:
    print("""
Audio files are ready!

Test them in the Streamlit app:
1. The app is running at: http://localhost:8501
2. Upload your audio files in the Detection tab
3. Click Analyze to see predictions
4. Use the threshold slider to adjust sensitivity
    """)
else:
    print("""
Audio files still needed. Choose an option:

Option B (Record yourself):
- Use Windows Sound Recorder
- Say "aaaa" or "oooo" for 3-5 seconds
- Save to: data/audio_samples/healthy_voice.wav
- Save to: data/audio_samples/parkinsons_voice.wav

Option C (Download datasets):
- Run: python scripts/download_datasets.py
- Download from provided sources
- Extract and place .wav files in data/audio_samples/

After adding files, run this script again to verify!
    """)

print("\n" + "=" * 70)
