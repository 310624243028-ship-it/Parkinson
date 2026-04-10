#!/usr/bin/env python
"""Diagnose audio file issues."""

import os
from pathlib import Path
import librosa
import wave

audio_file = Path('data/audio_samples/my_voice1.wav')

print("=" * 60)
print(f"Diagnosing: {audio_file}")
print("=" * 60)

if not audio_file.exists():
    print("❌ File not found!")
else:
    # Check file size
    size = audio_file.stat().st_size
    print(f"\n📊 File Info:")
    print(f"   Size: {size} bytes ({size/1024:.1f} KB)")
    
    if size < 1000:
        print("   ⚠️  File is too small - possibly empty!")
    
    # Try to read as WAV
    print(f"\n🔍 Checking WAV format...")
    try:
        with wave.open(str(audio_file), 'rb') as f:
            params = f.getparams()
            print(f"   ✓ Valid WAV file!")
            print(f"     - Channels: {params.nchannels}")
            print(f"     - Sample width: {params.sampwidth} bytes")
            print(f"     - Frame rate: {params.framerate} Hz")
            print(f"     - Frames: {params.nframes}")
            duration = params.nframes / params.framerate
            print(f"     - Duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"   ❌ Not a valid WAV: {e}")
    
    # Try to load with librosa
    print(f"\n📂 Trying librosa.load()...")
    try:
        y, sr = librosa.load(str(audio_file), sr=22050)
        print(f"   ✓ Successfully loaded!")
        print(f"     - Samples: {len(y)}")
        print(f"     - Sample rate: {sr} Hz")
        duration = len(y) / sr
        print(f"     - Duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"   ❌ Failed to load: {type(e).__name__}")
        print(f"      Message: {str(e)}")
        
        # Try alternative loading
        print(f"\n   Trying with sr=None...")
        try:
            y, sr = librosa.load(str(audio_file), sr=None)
            print(f"   ✓ Loaded with sr=None!")
            print(f"     - Samples: {len(y)}")
            print(f"     - Sample rate: {sr} Hz")
        except Exception as e2:
            print(f"   ❌ Also failed: {e2}")

print("\n" + "=" * 60)
