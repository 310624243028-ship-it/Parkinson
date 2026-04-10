# Parkinson's Disease Detection from Voice Recordings

This project uses machine learning to detect Parkinson's disease from breathing patterns in voice recordings.

## Features

- Audio feature extraction using Librosa
- Feature selection with Random Forest
- Classification using SVM
- Web UI with Streamlit

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a voice recording
2. The model will analyze breathing patterns
3. Get prediction results

## Project Structure

- `data/`: Dataset files
- `models/`: Trained models
- `scripts/`: Training and prediction scripts
- `app.py`: Streamlit application
- `utils.py`: Utility functions