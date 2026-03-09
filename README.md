The link for google colab notebook : https://colab.research.google.com/drive/19ayIueplobgoVohljGCeS9rmkUquysOz?usp=sharing

# AI-based Sleep Breathing Irregularity Detection

This project analyzes physiological sleep signals to detect breathing abnormalities such as apnea and hypopnea using machine learning and deep learning techniques.

The system processes overnight sleep recordings and builds a classification model to identify abnormal breathing patterns.

---

## Problem Statement

Sleep breathing disorders such as apnea can lead to serious health complications. Detecting these events automatically from physiological signals can help improve diagnosis and monitoring.

This project builds an end-to-end pipeline that:

1. Visualizes overnight physiological signals
2. Preprocesses and filters respiratory signals
3. Creates labeled datasets using annotated breathing events
4. Trains a deep learning model to classify breathing irregularities

---

## Dataset

Each participant contains 8 hours of sleep data with the following signals:

- Nasal Airflow (32 Hz)
- Thoracic Movement (32 Hz)
- SpO₂ – Oxygen Saturation (4 Hz)

Additional files include:

- Breathing event annotations (apnea / hypopnea)
- Sleep stage profile

---

## Project Pipeline

### 1. Data Visualization

The `vis.py` script visualizes physiological signals across the full sleep session.

Features:

- Plots Nasal Airflow
- Plots Thoracic Movement
- Plots SpO₂
- Overlays annotated breathing events

---

### 2. Signal Preprocessing

Respiration typically occurs in the range: 0.17 Hz – 0.4 Hz

Signals are filtered using digital filtering techniques to remove noise outside this range.

Steps performed:

- Bandpass filtering
- Signal synchronization using timestamps
- Window segmentation

---

### 3. Dataset Creation

Signals are divided into:
30 second windows
50% overlap


Labeling rule:

- If window overlaps >50% with apnea/hypopnea → label as event
- Otherwise → label as Normal

Dataset is stored as: Dataset/breathing_dataset.csv

---

### 4. Model Training

A **1D Convolutional Neural Network (CNN)** is trained to classify breathing irregularities from physiological signals.

Model input:

30-second signal windows

Evaluation strategy:

Leave-One-Participant-Out Cross Validation


In each fold:
- Train on 4 participants
- Test on 1 participant

---

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- Confusion Matrix


---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- TensorFlow / PyTorch
- Scikit-learn

---

## Future Improvements

- Add Conv-LSTM architecture
- Use attention-based time series models
- Improve feature extraction
- Expand dataset with more participants

---

## Author

Palak Mishra

