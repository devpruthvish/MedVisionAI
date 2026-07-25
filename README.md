# 🩺 Pneumonia Detection Using Deep Learning

## 📌 Project Overview

This project is a Deep Learning based web application that detects **Pneumonia** from Chest X-Ray images using **Transfer Learning with ResNet18**.

The application allows users to upload a chest X-ray image and predicts whether the patient is suffering from **Pneumonia** or is **Normal**.

The project was developed using **PyTorch** and deployed using **Streamlit**.

---

# ✨ Features

- Chest X-Ray Image Upload
- AI-based Pneumonia Detection
- Transfer Learning using ResNet18
- Confidence Score Display
- Class Probability Display
- Professional Streamlit User Interface
- Fast Image Prediction

---

# 📂 Dataset

Dataset Used:

Chest X-Ray Images

Classes:

- NORMAL
- PNEUMONIA

Dataset Structure:

```
chest_xray/
│
├── train/
├── val/
└── test/
```

---

# 🧠 Model Architecture

Model Used:

- ResNet18 (Transfer Learning)

Framework:

- PyTorch

Image Size:

- 224 × 224

Output Classes:

- NORMAL
- PNEUMONIA

---

# 📊 Model Performance

| Metric | Value |
|---------|--------|
| Test Accuracy | **81.25%** |
| Classes | NORMAL, PNEUMONIA |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |

---

# 🖥️ Web Application

The application provides:

- Upload Chest X-Ray
- Display Uploaded Image
- Prediction Result
- Confidence Score
- Probability for both Classes

---

# 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow
- NumPy
- Matplotlib

---

# 🚀 Installation

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

# 📷 Sample Output

The application predicts:

- NORMAL
- PNEUMONIA

along with confidence percentage and class probabilities.

---

# 📈 Future Improvements

- Improve model accuracy
- Fine-tune deeper ResNet layers
- Handle class imbalance
- Deploy on cloud
- Support DICOM images

---





