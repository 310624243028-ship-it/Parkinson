import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import os
from pathlib import Path
from utils import extract_features, preprocess_audio
import plotly.graph_objects as go
import plotly.express as px
import json
import time

# Set page config
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide"
)

# region agent log
_AGENT_DEBUG_LOG_PATH = str(Path(__file__).resolve().with_name("debug-d94e12.log"))
_AGENT_DEBUG_SESSION_ID = "d94e12"
def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        try:
            st.session_state["_agent_log_error"] = f"{type(e).__name__}: {e}"
        except Exception:
            pass
# endregion

_agent_log("H0", "app.py:startup", "App started", {"log_path": _AGENT_DEBUG_LOG_PATH})

# Title and description
st.title("🧠 Parkinson's Disease Detection")
st.markdown("Detect Parkinson's disease from voice recordings using machine learning")
st.markdown("---")

# Load pre-trained model
@st.cache_resource
def load_model():
    model_path = Path("models/parkinsons_svm_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    else:
        st.warning("Model not found. Please train the model first using scripts/train_model.py")
        return None

# Load scaler
@st.cache_resource
def load_scaler():
    scaler_path = Path("models/feature_scaler.pkl")
    if scaler_path.exists():
        return joblib.load(scaler_path)
    else:
        return None

# Load feature indices (selected features)
@st.cache_resource
def load_feature_indices():
    idx_path = Path("models/feature_indices.pkl")
    if idx_path.exists():
        try:
            return joblib.load(idx_path)
        except Exception:
            return None
    return None

# Load training data statistics
@st.cache_resource
def load_training_stats():
    data_path = Path("data/parkinsons.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        feature_cols = [col for col in df.columns if col != 'status' and col != 'name']
        
        healthy_data = df[df['status'] == 0][feature_cols]
        parkinsons_data = df[df['status'] == 1][feature_cols]
        
        stats = {
            'feature_names': feature_cols,
            'healthy': {
                'mean': healthy_data.mean().values,
                'std': healthy_data.std().values,
                'min': healthy_data.min().values,
                'max': healthy_data.max().values,
            },
            'parkinsons': {
                'mean': parkinsons_data.mean().values,
                'std': parkinsons_data.std().values,
                'min': parkinsons_data.min().values,
                'max': parkinsons_data.max().values,
            }
        }
        return stats
    return None

# Load training score distribution for threshold UI
@st.cache_resource
def load_training_score_stats():
    try:
        data_path = Path("data/parkinsons.csv")
        if not data_path.exists() or model is None or scaler is None or not hasattr(model, "weights"):
            return None
        df = pd.read_csv(data_path)
        X = df.drop(["status", "name"], axis=1, errors="ignore")
        y = df["status"] if "status" in df.columns else None
        X_scaled = scaler.transform(X)
        scores = X_scaled @ model.weights
        stats = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "p1": float(np.percentile(scores, 1)),
            "p99": float(np.percentile(scores, 99)),
        }
        if y is not None:
            stats["healthy_mean"] = float(np.mean(scores[y == 0]))
            stats["pd_mean"] = float(np.mean(scores[y == 1]))
        return stats
    except Exception:
        return None

# Main app
model = load_model()
scaler = load_scaler()
feature_indices = load_feature_indices()
score_stats = load_training_score_stats()

# Sidebar - Model threshold adjustment
with st.sidebar:
    st.header("⚙️ Model Settings")
    with st.expander("Debug (agent)", expanded=False):
        st.code(_AGENT_DEBUG_LOG_PATH)
        if "_agent_log_error" in st.session_state:
            st.error(st.session_state["_agent_log_error"])
    if model is not None:
        if hasattr(model, "threshold") and score_stats is not None:
            thr_min = float(np.floor(score_stats["p1"]))
            thr_max = float(np.ceil(score_stats["p99"]))
            thr_val = float(model.threshold)
            if thr_val < thr_min:
                thr_min = float(np.floor(thr_val))
            if thr_val > thr_max:
                thr_max = float(np.ceil(thr_val))
            new_threshold = st.slider(
                "Decision Threshold",
                min_value=thr_min,
                max_value=thr_max,
                value=thr_val,
                step=1.0,
                help="Lower threshold = more sensitive (more Parkinson's detections)\nHigher threshold = more conservative (fewer false positives)"
            )
            model.threshold = float(new_threshold)
        else:
            st.info("Threshold tuning not available (missing model threshold/weights or training score stats).")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detection", "📈 Feature Comparison", "📊 Dataset Info", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Voice Recording")
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])
        
        if audio_file is not None:
            # Save uploaded file temporarily
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Display audio player
            st.audio(audio_file)
            
            if st.button("🔬 Analyze", key="analyze_btn"):
                try:
                    # Load and process audio
                    with st.spinner("Processing audio..."):
                        try:
                            st.write("Loading audio file...")
                            y, sr = librosa.load("temp_audio.wav", sr=None)
                            duration_s = (len(y) / sr) if sr else 0.0
                            st.write(f"✓ Audio loaded: {len(y)} samples at {sr}Hz ({duration_s:.2f}s)")
                            _agent_log(
                                "H4",
                                "app.py:load_audio",
                                "Audio loaded",
                                {"samples": int(len(y)), "sr": int(sr) if sr is not None else None, "duration_s": float(duration_s), "dtype": str(getattr(y, "dtype", None))},
                            )

                            # Guard: extremely short clips produce unreliable features and default fallbacks
                            if duration_s < 1.0:
                                st.error(
                                    "This recording is too short for reliable analysis. "
                                    "Please upload at least 1–2 seconds of sustained voice (e.g., say 'aaaah')."
                                )
                                _agent_log(
                                    "H4",
                                    "app.py:duration_guard",
                                    "Rejected too-short audio",
                                    {"duration_s": float(duration_s)},
                                )
                                st.stop()
                        except Exception as e:
                            raise Exception(f"Failed to load audio: {str(e)}")
                        
                        try:
                            st.write("Extracting voice features...")
                            features = extract_features(y, sr)
                            st.write(f"✓ Extracted {len(features)} features")
                            _agent_log(
                                "H1",
                                "app.py:extract_features",
                                "Extracted features summary",
                                {
                                    "n_features": int(len(features)),
                                    "min": float(np.min(features)) if len(features) else None,
                                    "max": float(np.max(features)) if len(features) else None,
                                    "mean": float(np.mean(features)) if len(features) else None,
                                    "first5": [float(x) for x in features[:5]],
                                },
                            )
                        except Exception as e:
                            raise Exception(f"Feature extraction failed: {str(e)}")
                        
                        # Scale features
                        raw_vec = np.array(features, dtype=float)
                        if scaler is not None:
                            features_scaled_full = scaler.transform([raw_vec])[0]
                        else:
                            features_scaled_full = raw_vec

                        # Apply saved feature indices if present (training uses feature selection)
                        if feature_indices is not None:
                            try:
                                features_scaled = np.array([features_scaled_full[feature_indices]])
                            except Exception:
                                features_scaled = np.array([features_scaled_full])
                        else:
                            features_scaled = np.array([features_scaled_full])

                        _agent_log(
                            "H2",
                            "app.py:scale_and_select",
                            "Scaled/selected feature vector",
                            {
                                "scaler_loaded": bool(scaler is not None),
                                "feature_indices_loaded": bool(feature_indices is not None),
                                "scaled_len": int(features_scaled.shape[-1]),
                                "scaled_min": float(np.min(features_scaled)),
                                "scaled_max": float(np.max(features_scaled)),
                            },
                        )
                        
                        # Make prediction
                        if model is not None:
                            prediction = int(model.predict(features_scaled)[0])
                            prediction_prob = float(model.decision_function(features_scaled)[0])

                            _agent_log(
                                "H3",
                                "app.py:predict",
                                "Model prediction raw outputs",
                                {
                                    "model_type": str(type(model)),
                                    "classes_": [int(x) for x in getattr(model, "classes_", [])] if hasattr(model, "classes_") else None,
                                    "prediction": prediction,
                                    "decision_function": prediction_prob,
                                    "threshold_attr": float(getattr(model, "threshold")) if hasattr(model, "threshold") else None,
                                },
                            )
                            
                            # Store features in session for use in other tabs
                            st.session_state.last_features = features
                            st.session_state.last_prediction = prediction
                            
                            # Display results
                            st.success("Analysis Complete!")
                            
                            col_pred1, col_pred2 = st.columns(2)
                            
                            with col_pred1:
                                if prediction == 1:
                                    st.error("⚠️ Positive for Parkinson's signs detected")
                                else:
                                    st.success("✅ Negative - No Parkinson's signs detected")
                            
                            with col_pred2:
                                confidence = abs(prediction_prob)
                                st.metric("Confidence Score", f"{min(confidence * 100, 100):.1f}%")
                            
                            # More details
                            st.subheader("Feature Analysis")
                            feature_names = [
                                "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                                "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                                "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
                                "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
                            ]
                            
                            features_df = pd.DataFrame({
                                "Feature": feature_names[:len(features)],
                                "Value": features
                            })
                            st.dataframe(features_df, width='stretch')
                        else:
                            st.error("Model not available. Please train the model first.")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Troubleshooting tips:\n- Make sure the file is a valid WAV, MP3, or audio file\n- Try the pre-generated samples first\n- Re-record in Windows Sound Recorder with WAV format")
                finally:
                    # Clean up temporary file
                    if os.path.exists("temp_audio.wav"):
                        os.remove("temp_audio.wav")
        else:
            st.info("📁 Upload an audio file to get started")
    
    with col2:
        st.subheader("📋 Model Info")
        if model is not None:
            st.success("✅ Model Loaded")
            st.write(f"**Model Type:** SVM Classifier")
            st.write(f"**Features:** 22 voice characteristics")
            st.write(f"**Status:** Ready for predictions")
        else:
            st.error("❌ Model Not Loaded")

with tab2:
    st.subheader("� Feature Comparison Dashboard")
    
    training_stats = load_training_stats()
    
    if training_stats is None:
        st.error("Training data not available")
    else:
        if 'last_features' not in st.session_state:
            st.info("💡 Upload and analyze an audio recording first to see how your features compare to the training data!")
        else:
            features = st.session_state.last_features
            feature_names = training_stats['feature_names']
            
            st.success("✅ Your recording analyzed! See how it compares below.")
            
            # Create tabs for different feature categories
            feat_tab1, feat_tab2 = st.tabs(["📊 Individual Features", "📈 Overall Comparison"])
            
            with feat_tab1:
                st.write("**Feature ranges from training data with your recording highlighted:**")
                
                # Create comparison plots for each feature
                cols = st.columns(2)
                
                for idx, (feature_name, value) in enumerate(zip(feature_names, features)):
                    if idx % 2 == 0:
                        col = cols[0]
                    else:
                        col = cols[1]
                    
                    with col:
                        # Get statistics
                        healthy_mean = training_stats['healthy']['mean'][idx]
                        healthy_std = training_stats['healthy']['std'][idx]
                        parkinsons_mean = training_stats['parkinsons']['mean'][idx]
                        parkinsons_std = training_stats['parkinsons']['std'][idx]
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # X axis values for ranges
                        x_vals = [0, 1]
                        
                        # Healthy range (shaded area)
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[healthy_mean - healthy_std, healthy_mean - healthy_std],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[healthy_mean + healthy_std, healthy_mean + healthy_std],
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(0, 255, 0, 0.2)',
                            fill='tonexty',
                            name='Healthy Range',
                            hovertemplate="<b>Healthy Range</b><br>%{y:.4f}<extra></extra>"
                        ))
                        
                        # Healthy mean line
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[healthy_mean, healthy_mean],
                            mode='lines',
                            name='Healthy Mean',
                            line=dict(color='green', width=3, dash='solid'),
                            hovertemplate="<b>Healthy Mean</b><br>%{y:.4f}<extra></extra>"
                        ))
                        
                        # Parkinsons range (shaded area)
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[parkinsons_mean - parkinsons_std, parkinsons_mean - parkinsons_std],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[parkinsons_mean + parkinsons_std, parkinsons_mean + parkinsons_std],
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            fill='tonexty',
                            name='Parkinsons Range',
                            hovertemplate="<b>Parkinsons Range</b><br>%{y:.4f}<extra></extra>"
                        ))
                        
                        # Parkinsons mean line
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[parkinsons_mean, parkinsons_mean],
                            mode='lines',
                            name='Parkinsons Mean',
                            line=dict(color='red', width=3, dash='solid'),
                            hovertemplate="<b>Parkinsons Mean</b><br>%{y:.4f}<extra></extra>"
                        ))
                        
                        # Your recording value (thick blue line)
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=[value, value],
                            mode='lines',
                            name='Your Recording',
                            line=dict(color='blue', width=4, dash='dash'),
                            hovertemplate="<b>Your Recording</b><br>%{y:.4f}<extra></extra>"
                        ))
                        
                        fig.update_layout(
                            title=f"<b>{feature_name}</b>",
                            height=350,
                            showlegend=True,
                            yaxis_title="Value",
                            xaxis=dict(showticklabels=False, showgrid=False),
                            hovermode='y unified',
                            margin=dict(l=60, r=20, t=60, b=40),
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=0.99,
                                xanchor="right",
                                x=0.99,
                                bgcolor="rgba(255,255,255,0.7)"
                            )
                        )
                        
                        st.plotly_chart(fig, width='stretch')
            
            with feat_tab2:
                st.write("**Overall feature profile comparison:**")
                
                # Normalize features for better visualization
                healthy_norm = (training_stats['healthy']['mean'] - training_stats['healthy']['min']) / (training_stats['healthy']['max'] - training_stats['healthy']['min'] + 1e-8)
                parkinsons_norm = (training_stats['parkinsons']['mean'] - training_stats['parkinsons']['min']) / (training_stats['parkinsons']['max'] - training_stats['parkinsons']['min'] + 1e-8)
                user_norm = (features - training_stats['healthy']['min']) / (training_stats['healthy']['max'] - training_stats['healthy']['min'] + 1e-8)
                
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=healthy_norm,
                    theta=feature_names,
                    fill='toself',
                    name='Healthy',
                    line=dict(color='green'),
                    opacity=0.6
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=parkinsons_norm,
                    theta=feature_names,
                    fill='toself',
                    name='Parkinsons',
                    line=dict(color='red'),
                    opacity=0.6
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=user_norm,
                    theta=feature_names,
                    fill='toself',
                    name='Your Recording',
                    line=dict(color='blue', width=3),
                    opacity=0.8
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    height=600,
                    title="<b>Feature Profile Radar Chart</b>"
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Add detailed statistics table
                st.write("**Detailed Feature Statistics:**")
                
                comparison_data = []
                for i, feature in enumerate(feature_names):
                    comparison_data.append({
                        'Feature': feature,
                        'Your Value': f"{features[i]:.4f}",
                        'Healthy Mean': f"{training_stats['healthy']['mean'][i]:.4f}",
                        'Healthy Std': f"{training_stats['healthy']['std'][i]:.4f}",
                        'Parkinsons Mean': f"{training_stats['parkinsons']['mean'][i]:.4f}",
                        'Parkinsons Std': f"{training_stats['parkinsons']['std'][i]:.4f}",
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch')

with tab3:
    st.subheader("📊 Dataset Information")
    
    # Load sample data
    data_path = Path("data/parkinsons.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            healthy = (df['status'] == 0).sum()
            st.metric("Healthy Samples", healthy)
        with col3:
            parkinsons = (df['status'] == 1).sum()
            st.metric("Parkinson's Samples", parkinsons)
        
        st.markdown("---")
        st.write("**Dataset Preview:**")
        st.dataframe(df.head(10), width='stretch')
        
        st.markdown("---")
        st.write("**Feature Descriptions:**")
        feature_descriptions = {
            "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
            "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
            "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
            "MDVP:Jitter(%)": "Measures of variation in fundamental frequency",
            "MDVP:Shimmer": "Measures of variation in amplitude",
            "NHR": "Noise-to-harmonics ratio",
            "HNR": "Harmonics-to-noise ratio",
            "RPDE": "Recurrence period density entropy",
            "DFA": "Detrended fluctuation analysis",
            "status": "Target variable (0=Healthy, 1=Parkinson's)"
        }
        
        for feature, desc in feature_descriptions.items():
            st.write(f"• **{feature}:** {desc}")
    else:
        st.warning("Dataset file not found")

with tab4:
    st.subheader("ℹ️ About This Application")
    
    st.write("""
    ### Project Overview
    This application uses machine learning to detect Parkinson's disease from voice recordings. 
    Parkinson's disease often affects voice quality, and analysis of voice characteristics can 
    help in early detection.
    
    ### How It Works
    1. **Audio Processing:** Voice recordings are processed using Librosa
    2. **Feature Extraction:** 22 voice characteristics are extracted
    3. **Normalization:** Features are normalized using a pre-fitted scaler
    4. **Classification:** SVM classifier predicts presence of Parkinson's disease
    
    ### Model Performance
    - **Algorithm:** Support Vector Machine (SVM)
    - **Feature Selection:** Random Forest importance-based
    - **Training Data:** Labeled voice recordings from multiple subjects
    
    ### Features Analyzed
    - Fundamental frequency variations (Jitter, Shimmer)
    - Spectral characteristics (NHR, HNR)
    - Nonlinear dynamics (RPDE, DFA, D2)
    - Voice quality metrics (PPE, spread1, spread2)
    
    ### Disclaimer
    This application is for educational and research purposes. 
    It should not be used as a substitute for professional medical diagnosis. 
    Always consult with qualified healthcare professionals.
    
    ### Project Structure
    ```
    Parkinsons/
    ├── app.py                    # Main Streamlit application
    ├── utils.py                  # Utility functions
    ├── requirements.txt          # Project dependencies
    ├── README.md                 # Project documentation
    ├── data/
    │   ├── parkinsons.csv        # Training dataset
    │   └── parkinsons_test.csv   # Test dataset
    ├── models/
    │   ├── parkinsons_svm_model.pkl      # Trained SVM model
    │   └── feature_scaler.pkl            # Feature scaler
    └── scripts/
        ├── train_model.py        # Model training script
        └── evaluate_model.py     # Model evaluation script
    ```
    """)

    st.markdown("---")
    st.write("**Built with:** Streamlit, Scikit-learn, Librosa, NumPy, Pandas")
