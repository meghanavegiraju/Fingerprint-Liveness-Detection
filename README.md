# Fingerprint-Liveness-Detection

## Project Overview

This project implements an AI-based Fingerprint Liveness Detection system that classifies a fingerprint image as either **LIVE** or **SPOOF** using a deep learning model.

The project is built using **PyTorch**, **MobileNetV3-Small**, and **Streamlit**.

---

## Features

- Fingerprint image preprocessing
- Binary classification (LIVE / SPOOF)
- Transfer Learning using MobileNetV3-Small
- Model evaluation using:
  - APCER
  - BPCER
  - Equal Error Rate (EER)
- Score Distribution Plot
- APCER vs BPCER Curve
- Image inference with confidence score
- Streamlit web application

---

## Project Structure

```
Fingerprint-Liveness-Detection/
│
├── app.py
├── preprocess.py
├── requirements.txt
├── README.md
├── Report.docx
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── liveness_train.py
│   ├── liveness_eval.py
│   ├── liveness_infer.py
│   └── liveness_model.pth
│
└── outputs/
    ├── score_distribution.png
    └── apcer_bpcer_curve.png
```

---

## Dataset

- LIVE Images: 30
- SPOOF Images: 30

Image Size:

```
224 × 224
```

Dataset Split:

- Training: 70%
- Validation: 15%
- Testing: 15%

---

## Model

CNN Backbone:

- MobileNetV3-Small

Optimizer:

- Adam

Learning Rate:

```
0.001
```

Epochs:

```
20
```

Classes:

- LIVE
- SPOOF

---

## Evaluation Metrics

The project evaluates the model using:

- APCER
- BPCER
- Equal Error Rate (EER)

Generated plots:

- Score Distribution
- APCER vs BPCER Curve

---

## Installation

Clone the repository:

```bash
git clone https://github.com/meghanavegiraju/Fingerprint-Liveness-Detection.git
```

Move into the project folder:

```bash
cd Fingerprint-Liveness-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Train Model

```bash
python models/liveness_train.py
```

### Evaluate Model

```bash
python models/liveness_eval.py
```

### Test a Single Image

```bash
python models/liveness_infer.py --image "dataset/test/live/D-10.jpeg"
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

Open the browser:

```
http://localhost:8501
```

Upload a fingerprint image to classify it as:

- ✅ LIVE
- ❌ SPOOF

The application also displays the prediction confidence.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

---

## Future Improvements

- Larger fingerprint dataset
- More spoof attack types
- Improved threshold calibration
- Real-time webcam inference
- Enhanced fingerprint quality assessment

---

## Author

**Meghana Vegiraju**

B.Tech – Artificial Intelligence and Data Science

---

## License

This project is developed for educational and assignment purposes.
