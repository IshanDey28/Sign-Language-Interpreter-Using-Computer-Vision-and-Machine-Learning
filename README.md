# 🤟 Sign Language Interpreter

> Real-time sign language recognition using computer vision and machine learning.

## 📖 About

Sign Language Interpreter is a real-time computer vision project that recognizes selected static sign-language alphabet gestures through a webcam.

The system uses OpenCV to capture webcam frames, MediaPipe Hands to detect 21 hand landmarks, and a Random Forest classifier to recognize the corresponding sign. The 21 landmarks provide X, Y, and Z coordinates, resulting in 63 numerical features for each sample.

A custom dataset is used to train the machine-learning model. During real-time operation, the system predicts the sign, displays its confidence, and uses a temporal buffer to make predictions more stable.

## ✨ Features

- Real-time webcam-based hand tracking
- 21-point hand landmark detection
- 63 numerical hand features
- Custom sign-language dataset
- Random Forest classification
- Real-time prediction and confidence score
- Temporal prediction buffering
- Live hand-landmark visualization

## 🔤 Supported Signs

The current model recognizes:

**A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S**

J and Z are not currently included because they involve dynamic hand movements rather than a single static hand pose.

## 🔄 How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hands
   ↓
21 Hand Landmarks
   ↓
63 Features (X, Y, Z)
   ↓
Random Forest Classifier
   ↓
Prediction + Confidence
   ↓
Temporal Buffer
   ↓
Confirmed Sign