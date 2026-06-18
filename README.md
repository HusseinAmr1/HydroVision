# 🌊 HydroVision
## AI-Based Flood Prediction Using Satellite & Climate Data

HydroVision is an Artificial Intelligence-powered flood prediction platform that integrates satellite imagery and climate data to support flood monitoring, risk assessment, and disaster preparedness.

The system combines Deep Learning, Machine Learning, Remote Sensing, and Climate Analytics to identify flooded regions, assess flood risks, and forecast future flood events.

---

## 📌 Project Overview

Floods are among the most destructive natural disasters worldwide, causing severe economic losses, environmental damage, and threats to human life.

HydroVision addresses this challenge by leveraging:

- Sentinel-1 SAR Satellite Imagery
- Sentinel-2 Optical Satellite Imagery
- ERA5 Climate Reanalysis Data
- Artificial Intelligence Models

The platform analyzes environmental conditions and generates flood predictions through an end-to-end AI pipeline.

---

## 🏗️ System Architecture

Data Collection
       ↓
Data Preprocessing
       ↓
Feature Extraction
       ↓
Flood Segmentation
       ↓
Risk Prediction
       ↓
Temporal Forecasting
       ↓
Result Visualization
       ↓
Decision Support

🚀 Key Features
Flood Segmentation

Detect flooded regions from satellite imagery using Deep Learning models.

Flood Risk Assessment

Estimate flood probability using environmental and climate-related variables.

Temporal Forecasting

Predict future flood risks based on historical climate patterns.

Interactive Dashboard

Visualize predictions, risk levels, and flood maps through a user-friendly interface.

Data Processing Pipeline

Automated preprocessing, synchronization, augmentation, and feature extraction workflows.

🤖 AI Models
Deep Learning
U-Net
ResNet-50 Backbone
ResNet-101 Backbone
Machine Learning
Gradient Boosting Classifier
AdaBoost Classifier
🛠️ Technologies Used
Programming Language
Python 3.10+
AI & Machine Learning
PyTorch
Scikit-Learn
Segmentation Models PyTorch
Torchvision
Backend
FastAPI
Frontend
Streamlit
Database
SQLite
SQLAlchemy
Data Processing
NumPy
Pandas
SciPy
Tifffile
Visualization
Matplotlib
Testing
Pytest
Version Control
Git
GitHub

📦 Python Libraries
numpy
pandas
scipy
torch
torchvision
segmentation-models-pytorch
timm
scikit-learn
joblib
tifffile
pyproj
albumentations
opencv-python-headless
matplotlib
uvicorn
sqlalchemy
passlib
python-jose
python-multipart
xarray
netCDF4
📊 Dataset
Satellite Data
Sentinel-1 (SAR)
Sentinel-2 (Optical)
Climate Data
ERA5 Reanalysis Dataset
Dataset Statistics
Description	Count
Initial Images	20,711
Filtered Images	13,434

The filtering process removed duplicate and unsynchronized samples to prevent data leakage and improve model reliability.

📈 Results
Flood Segmentation
Metric	Value
Accuracy	89.75%
mIoU	64.10%
Flood Risk Prediction
Metric	Value
ROC AUC	76.61%
Temporal Forecasting
Metric	Value
ROC AUC	88.24%
📂 Project Structure
HydroVision/
│
├── data/
├── models/
├── notebooks/
├── api/
├── dashboard/
├── tests/
├── outputs/
├── app.py
├── web_api.py
├── requirements.txt
└── README.md
🔬 Research Contributions
Integration of satellite imagery and climate data.
Development of an AI-driven flood prediction framework.
Implementation of flood segmentation and risk assessment modules.
Temporal flood forecasting using environmental observations.
Interactive visualization and decision support platform.
👨‍💻 Team Members
Hussein Amr Shaaban Rabie
Ahmed Moustafa Kamel Ali
Naima Gamal Mostafa
Ahella Hesham Ahmed Elsadek
Manar Adel Fawzy Amin
Mohamed Abdelaty Abdelrahman Mohamed
👩‍🏫 Supervisor

Dr. Eman Monir

📜 License

This project was developed as a Bachelor of Computer Science Graduation Project (2025–2026).

⭐ Acknowledgment

Special thanks to Dr. Eman Monir for her guidance, support, and valuable feedback throughout the development of this project.
