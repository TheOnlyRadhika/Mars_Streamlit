#  Deepfake Audio Detection

A deep learning-based system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using **Mel-Frequency Cepstral Coefficients (MFCCs)** and Convolutional Neural Networks (CNNs).

---

##  Problem Statement

Recent advances in generative AI have made it possible to synthesize highly realistic speech that closely mimics human voices. Such deepfake audio poses serious risks including misinformation, fraud, and identity impersonation.

This project aims to develop a robust binary classifier capable of determining whether an audio recording is:

-  **Real (Human Speech)**
-  **Fake (AI-Generated Speech)**

---

##  Project Structure

```text
deepfake-audio-detection/
│
├── app.py                         
├── test_model.py                 
├── best_cnn_model.keras          
├── deepfake_audio_detection.ipynb 
├── pyproject.toml                 
└── README.md
```

---

##  Dataset

### Fake-or-Real Dataset

Dataset Source:

https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

The project uses the **for-norm** variant containing normalized audio recordings.

| Split      |   Real |   Fake |  Total |
| ---------- | -----: | -----: | -----: |
| Train      | 26,941 | 26,927 | 53,868 |
| Validation |  5,400 |  5,398 | 10,798 |
| Test       |  2,264 |  2,370 |  4,634 |

### Dataset Observations

- Classes are nearly perfectly balanced.
- All recordings are sampled at **16 kHz**.
- No class rebalancing techniques were required.

---

## Methodology

### Audio Preprocessing Pipeline

```text
Audio File
    ↓
Load & Validate
    ↓
Resample to 16 kHz
    ↓
Convert to Mono
    ↓
Remove DC Offset
    ↓
Apply Pre-emphasis Filter
    ↓
Normalize Amplitude
    ↓
Trim Silence
    ↓
Fix Duration (5 seconds)
    ↓
Extract 40 MFCC Features
    ↓
Standardize Features
    ↓
Fix Shape (40 × 156)
    ↓
Add Channel Dimension
    ↓
CNN Input
```

---

### Feature Extraction

The audio signal is converted into MFCC representations.

Configuration:

- Number of MFCCs: **40**
- Fixed feature shape: **40 × 156 × 1**
- Hop Length: **512**
- Input format: Grayscale feature map

These MFCC feature maps are treated similarly to grayscale images and supplied as input to the CNN.

---

##  Models

### Model 1 — CNN (Selected Model)

Architecture:

- Conv2D + Batch Normalization
- Max Pooling
- Dropout
- Conv2D + Batch Normalization
- Max Pooling
- Dropout
- Conv2D + Batch Normalization
- Max Pooling
- Dropout
- Flatten
- Dense (128)
- Sigmoid Output Layer

---

### Model 2 — LSTM

Architecture:

- LSTM (128 units)
- LSTM (64 units)
- Dense Layer
- Batch Normalization
- Sigmoid Output Layer

---

##  Results

### CNN (Selected Model)

| Metric        |  Value | Target |
| ------------- | -----: | -----: |
| Test Accuracy | 92.73% |  ≥ 80% |
| F1 Score      | 93.30% |  ≥ 80% |
| EER           |  4.38% |  ≤ 12% |
| Real Accuracy | 86.17% |  ≥ 75% |
| Fake Accuracy | 98.99% |  ≥ 75% |

---

### LSTM

| Metric        |  Value | Target |
| ------------- | -----: | -----: |
| Test Accuracy | 88.86% |  ≥ 80% |
| F1 Score      | 89.13% |  ≥ 80% |
| EER           | 11.16% |  ≤ 12% |
| Real Accuracy | 88.43% |  ≥ 75% |
| Fake Accuracy | 89.28% |  ≥ 75% |

---

##  Model Selection

The CNN was selected as the final model because it consistently outperformed the LSTM across all evaluation metrics.

### Performance Comparison

| Metric   |    CNN |   LSTM |
| -------- | -----: | -----: |
| Accuracy | 92.73% | 88.86% |
| F1 Score | 93.30% | 89.13% |
| EER      |  4.38% | 11.16% |

Key observations:

- CNN achieved the highest overall accuracy.
- CNN reduced Equal Error Rate (EER) by more than 60%.
- CNN achieved near-perfect fake audio detection (98.99%).

---

##  Running the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### Install Dependencies

```bash
pip install streamlit librosa numpy tensorflow scikit-learn
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

### Test a Single Audio File

```bash
python test_model.py --file "/path/to/audio.wav"
```

---

##  Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Seaborn

---

##  Reproducing Training

1. Download the dataset from Kaggle.
2. Open `deepfake_audio_detection.ipynb`.
3. Run all notebook cells sequentially.
4. Train the CNN and LSTM models.
5. The best-performing CNN model will be saved as:

```text
best_cnn_model.keras
```

---

##  License

This project is intended for educational and research purposes.
