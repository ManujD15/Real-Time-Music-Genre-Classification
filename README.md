# Real Time Music Genre Classification
This project uses Machine Learning (ML) and Deep Learning (DL) techniques to classify music into multiple genres using the **GTZAN Dataset**. The key feature is a **graphical user interface (GUI)** for real-time genre predictions.

## Table of Contents

- [Methodology](#methodology)
- [Project Setup](#project-setup)
- [Directory Structure](#directory-structure)
- [Instructions](#instructions)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Resources](#resources)

---

## Methodology

### 1. Machine Learning
Key audio features were extracted using the `Librosa` library, including:
- **Mel Frequency Cepstral Coefficients (MFCC)**
- Spectral Centroid, Flux, Contrast, Bandwidth
- Zero-Crossing Rate, Tempo, and Loudness

The extracted features were used to train:
- **Support Vector Machines (SVM)**
- **Extreme Gradient Boosting (XGBoost)**

### 2. Deep Learning
Spectrogram images derived from audio files were used as input for:
- **Convolutional Neural Networks (CNNs)**

Two CNN architectures were implemented:
- **Initial CNN Model**: Focused on spectrograms for feature learning.
- **CNN 2**: Improved accuracy and mitigated bias observed in the first model.

---

## Project Setup

### Dependencies
Ensure the following packages are installed:
- `TensorFlow`
- `Keras`
- `Scikit-learn`
- `Librosa`
- `Matplotlib`
- `Tkinter`
- `Python 3.8+`

Install dependencies via `pip`:
```bash
pip install -r requirements.txt

----

## Dataset
Download the GTZAN Dataset from GTZAN.
The dataset contains 1,000 audio tracks across 10 genres: classical, blues, pop, disco, jazz, metal, reggae, country, rock, and hip-hop.
