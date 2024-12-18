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
```

Dataset
Download the GTZAN Dataset from GTZAN.
The dataset contains 1,000 audio tracks across 10 genres: classical, blues, pop, disco, jazz, metal, reggae, country, rock, and hip-hop.




Here’s the full README.md content for your project:

markdown
Copy code
# Real-Time Music Genre Classification

This repository contains the code and methodology for **Real-Time Music Genre Classification**, a project conducted as part of a Master's thesis in Data Analytics. It employs Machine Learning (ML) and Deep Learning (DL) techniques to classify music genres using the **GTZAN Dataset**. A key feature is a **graphical user interface (GUI)** for real-time genre predictions.

---

## Table of Contents

- [Methodology](#methodology)
- [Project Setup](#project-setup)
- [Directory Structure](#directory-structure)
- [Instructions](#instructions)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Resources](#resources)
- [License](#license)

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
```

Dataset
Download the GTZAN Dataset from GTZAN.
The dataset contains 1,000 audio tracks across 10 genres: classical, blues, pop, disco, jazz, metal, reggae, country, rock, and hip-hop.

Directory Structure
/data/: Contains raw and preprocessed data.
/models/: Pretrained models (e.g., manuj_cnn.h5).
/notebooks/: Jupyter notebooks for training and evaluation.
feature_extraction.ipynb: Extracts features for ML models.
train_svm_xgboost.ipynb: Implements ML models.
train_cnn.ipynb: Trains the CNN.
/src/:
realtime_classification.py: GUI-based real-time classification.
data_visualization.py: Visualizes audio features.
train_cnn2.py: Trains the enhanced CNN 2.

Instructions
1. Preprocessing
Extract features and prepare datasets:

bash
Copy code
python src/feature_extraction.py
2. Train Models
Train SVM/XGBoost
bash
Copy code
python src/train_svm_xgboost.py
Train CNN
bash
Copy code
python src/train_cnn.py
3. Real-Time Classification
Run the GUI for real-time music genre predictions:

bash
Copy code
python src/realtime_classification.py
Results
Model Performances
SVM: 76% accuracy
XGBoost: 69% accuracy
CNN: 76.6% accuracy
CNN 2: 76.37% accuracy with improved bias handling
Confusion Matrices and Accuracy Graphs
Results and visualizations are available in the /results/ folder.

Future Enhancements
Dataset Expansion: Incorporate modern datasets to improve genre diversity.
Data Augmentation: Add techniques such as pitch shifting, time stretching, and noise addition.
Ensemble Learning: Combine predictions from multiple models (e.g., SVM, XGBoost, CNN).
GUI Optimization: Address latency and enhance real-time resource handling.
Continuous Model Updates: Retrain models periodically with new data to adapt to emerging genres.
Resources
GTZAN Dataset
Pretrained CNN Model
