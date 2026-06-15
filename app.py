import streamlit as st
import numpy as np
import librosa
import tempfile
import os
from keras.models import load_model

# ── Constants (same as notebook) ─────────────────────────────────────────────
TARGET_SR   = 16000
DURATION    = 5
MAX_LEN     = TARGET_SR * DURATION
HOP_LENGTH  = 512
N_MFCC      = 40
MAX_FRAMES  = MAX_LEN // HOP_LENGTH

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    model = load_model("best_cnn_model.keras")
    return model

model = load_cnn_model()

# ── Preprocessing Pipeline (same as notebook) ─────────────────────────────────
def preprocess_audio(audio, sr):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    audio = audio - np.mean(audio)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    audio, _ = librosa.effects.trim(audio, top_db=20)
    return audio

def fix_duration(audio):
    if len(audio) < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))
    else:
        audio = audio[:MAX_LEN]
    return audio

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
           (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc[..., np.newaxis]

def predict(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    audio     = preprocess_audio(audio, sr)
    audio     = fix_duration(audio)
    mfcc      = extract_mfcc(audio)
    mfcc      = np.expand_dims(mfcc, axis=0)  # add batch dimension
    prob      = model.predict(mfcc, verbose=0)[0][0]
    label     = "Deepfake (AI-Generated)" if prob > 0.5 else "Genuine (Human)"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, prob

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

st.title("🎙️ Deepfake Audio Detector")
st.markdown("Upload a speech recording to detect whether it is **Genuine (Human)** or **Deepfake (AI-Generated)**.")
st.divider()

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("Analyzing audio..."):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Predict
        label, confidence, prob = predict(tmp_path)
        os.remove(tmp_path)
    
    st.divider()
    
    # Result
    if "Genuine" in label:
        st.success(f"✅ **{label}**")
    else:
        st.error(f"🚨 **{label}**")
    
    # Confidence
    st.metric("Confidence Score", f"{confidence*100:.2f}%")
    
    # Probability bar
    st.markdown("**Prediction Probability**")
    col1, col2 = st.columns(2)
    col1.metric("Genuine", f"{(1-prob)*100:.2f}%")
    col2.metric("Deepfake", f"{prob*100:.2f}%")
    
    st.progress(float(prob))

st.divider()
st.caption("Built with CNN + MFCC features | Fake-or-Real Dataset")