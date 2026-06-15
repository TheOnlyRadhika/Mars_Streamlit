import argparse
import numpy as np
import librosa
from keras.models import load_model
import os

# ── Constants (same as notebook) ─────────────────────────────────────────────
TARGET_SR  = 16000
DURATION   = 5
MAX_LEN    = TARGET_SR * DURATION
HOP_LENGTH = 512
N_MFCC     = 40
MAX_FRAMES = MAX_LEN // HOP_LENGTH

# ── Load Model ────────────────────────────────────────────────────────────────
model = load_model("best_cnn_model.keras")

# ── Preprocessing Pipeline ────────────────────────────────────────────────────
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
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc[..., np.newaxis]

def predict(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    audio, sr  = librosa.load(file_path, sr=None)
    audio      = preprocess_audio(audio, sr)
    audio      = fix_duration(audio)
    mfcc       = extract_mfcc(audio)
    mfcc       = np.expand_dims(mfcc, axis=0)
    prob       = model.predict(mfcc, verbose=0)[0][0]
    label      = "Deepfake (AI-Generated)" if prob > 0.5 else "Genuine (Human)"
    confidence = prob if prob > 0.5 else 1 - prob

    print("=" * 45)
    print("       DEEPFAKE AUDIO DETECTION RESULT")
    print("=" * 45)
    print(f"  File       : {os.path.basename(file_path)}")
    print(f"  Result     : {label}")
    print(f"  Confidence : {confidence*100:.2f}%")
    print(f"  Genuine    : {(1-prob)*100:.2f}%")
    print(f"  Deepfake   : {prob*100:.2f}%")
    print("=" * 45)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection")
    parser.add_argument("--file", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()
    predict(args.file)