# 🎙️ SENSE AI: Speech Emotion Analyzer (v2.2)

![Python](https://img.shields.io/badge/Python-3.7-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1.14.0-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.0.6-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.2.5-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **"Bridging the gap between human emotion and artificial intelligence."**

SENSE AI is a high-fidelity Speech Emotion Recognition (SER) platform powered by deep learning. It uses a Convolutional Neural Network (CNN) to classify human emotions from audio and video signals with over 70% validation accuracy. This project is designed for researchers, developers, and final-year students looking for a production-ready sentiment analysis pipeline.

---

## 🌟 Modern Interactive Dashboard (SENSE AI v2.2)

Our latest update introduces a **Mix-Media Intelligence Dashboard** that brings the AI to life with a stunning, reactive interface.

![](images/sense_ai_gui.png?raw=true)

### 💎 Key Features:
*   **🎭 Reactive UI Theme**: The entire dashboard dynamically shifts its color palette (Angry/Red, Happy/Gold, Calm/Green) based on the detected emotional vector.
*   **🎬 Mixed Media Support**: Effortlessly extract emotions from `.wav`, `.mp3` audio, and `.mp4`, `.mov` video files using automated track stripping.
*   **🎤 Live Voice Capture**: Record directly from your microphone with real-time timers and animated signal pulses.
*   **📊 Deep Signal Metrics**: Instant extraction of **Vocal Tempo (BPM)**, **Mean Pitch (Hz)**, and **Vocal Energy (%)**.
*   **📜 Neural Reports**: Download professional, printable **PDF Reports** containing confidence bars, AI insights, and vocal metrics.
*   **〰️ Waveform Visualizer**: Interactive signal visualization for every processed file.

---

## 🔬 Technical Methodology

### 1. Data Engineering
We utilized the industry-standard **RAVDESS** and **SAVEE** datasets, providing a balanced exposure to diverse emotional archetypes across both male and female speakers.

### 2. Signal Analysis & Feature Extraction
We employ **Mel-Frequency Cepstral Coefficients (MFCCs)** via the `Librosa` library. This captures the power spectrum of audio signals, mimicking human auditory perception.
*   **Frequency Doubling**: We double the sampling rate to capture higher-resolution features.
*   **3-Second Normalization**: All signals are windowed to 3s for input consistency.

![](images/feature.png?raw=true)

### 3. Deep Learning Architecture
The core is a **Convolutional Neural Network (CNN)** optimized for 1D signal classification. 
*   **Layers**: Conv1D → MaxPooling → Dropout → Dense.
*   **Training**: Optimized via categorical cross-entropy and RMSProp.
*   **Accuracy**: Achieved **70%+ validation accuracy** across 10 emotional classes.

![](images/cnn.png?raw=true)

---

## 💡 Emotion Mapping
The model classifies signals into 10 distinct categories, mapping gender and sentiment with high precision:

| Code | Emotion Label | Code | Emotion Label |
| :--- | :--- | :--- | :--- |
| **0** | Female Angry | **5** | Male Angry |
| **1** | Female Calm | **6** | Male Calm |
| **2** | Female Fearful | **7** | Male Fearful |
| **3** | Female Happy | **8** | Male Happy |
| **4** | Female Sad | **9** | Male Sad |

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.7 (Recommended for TF 1.14 compatibility)
*   Virtual Environment (venv)

### Installation & Run
1.  **Clone & Navigate**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/your-repo.git
    cd Speech-Emotion-Analyzer-master
    ```
2.  **Activate Environment**:
    ```bash
    source .venv/bin/activate
    ```
3.  **Install Engine Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch SENSE AI**:
    ```bash
    python3 app.py
    ```
5.  **Access GUI**: Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.

---

## 🚀 Deployment (GitHub Actions + Render)

We have configured a CI/CD pipeline (`.github/workflows/render_deploy.yml`) that automatically builds the Docker image on GitHub's fast servers and pushes it to your repository packages.

### Steps to Activate:
1.  **Get Deploy Hook**:
    *   In your Render Web Service -> Settings -> **Deploy Hook**.
    *   Copy the URL.
2.  **Add Secret**:
    *   Go to your GitHub Repo -> Settings -> Secrets and variables -> Actions -> **New Repository Secret**.
    *   Name: `RENDER_DEPLOY_HOOK`
    *   Value: (Paste the URL).
3.  **Update Render Image Source**:
    *   In Render -> Settings -> **Image URL**.
    *   Change it to: `ghcr.io/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME:latest` (lowercase).
    *   Now, every push to `main` will automatically build and deploy!

---

## 🗺️ Future Roadmap
*   [ ] **Real-time VU Meter**: Live visual frequency bars during recording.
*   [ ] **Translation Sync**: Multi-language speech-to-text integration.
*   [ ] **Stress Metrics**: Physiological stress level estimation from vocal jitter.

---

## 📜 Conclusion
Building **SENSE AI** involved extensive hyperparameter tuning and signal normalization. The resulting model distinguishes between male and female voices with nearly **100% accuracy** and interprets nuanced emotional signatures with over **70% accuracy**, making it a robust foundation for next-gen sentiment analysis.

---
**Developed for Final Year Project Showcase.** 
*Licensed under MIT.*
