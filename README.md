<h1 align="center">🖐️ Hand Gesture Image Recognition System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-success?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/cvzone-MediaPipe-informational?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

---

## 🎯 Project Overview

The **Hand Gesture Image Recognition System** is a real-time data collection tool that leverages **OpenCV**, **MediaPipe** (via `cvzone`), and **NumPy** to detect, crop, and standardize hand gesture images from webcam input.

> 📦 It allows you to build a clean and structured dataset for training ML/DL models in gesture recognition, sign language classification, or HCI systems.

<p align="center">
  <img src="https://user-images.githubusercontent.com/0000000/preview-image.gif" alt="Demo GIF" width="600"/>
</p>

---

## 💡 Key Features

✅ Real-time hand detection using **cvzone.HandTrackingModule**  
✅ Smart cropping with dynamic **padding + aspect-ratio preserving resize**  
✅ Outputs 300×300 pixel centered gesture images  
✅ Easy hotkey-based image saving (press `s`)  
✅ Lightweight and optimized for **custom dataset creation**

---

## 🧠 Use Cases

- ✋ Sign language dataset generation
- 🤖 AI model training for gesture classification
- 🕹️ Gesture-based UI/UX prototypes
- 🎮 Gesture-controlled games or robots

---

## 🧰 Tech Stack

| Technology | Usage |
|-----------|--------|
| ![Python](https://img.shields.io/badge/-Python-333?logo=python) | Core Programming Language |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-333?logo=opencv) | Image Processing |
| ![cvzone](https://img.shields.io/badge/-cvzone-333) | Wrapper for MediaPipe Hand Tracking |
| ![NumPy](https://img.shields.io/badge/-NumPy-333?logo=numpy) | Array & image manipulation |

---

## 🧪 Demo

> 📽️ Live hand detection, cropping, and saving upon key press:

<img src="https://user-images.githubusercontent.com/0000000/demo.gif" width="600" alt="Demo of hand gesture image capture" />

---

## ⚙️ How It Works

```mermaid
graph TD;
    A[Webcam Input] --> B[Hand Detection (cvzone)]
    B --> C[Bounding Box Extraction]
    C --> D[Crop + Padding]
    D --> E[Resize with Aspect Ratio]
    E --> F[300x300 Image Preview]
    F --> G[Save to Dataset on Key Press]
