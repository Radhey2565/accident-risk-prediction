# Accident Risk Prediction System
> AI-Powered Road Safety Risk Analysis using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-81%25-brightgreen)]()
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://radhey2565-accident-risk-prediction-appapp-rykbze.streamlit.app/)

An intelligent Machine Learning system that predicts road accident risk in real-time based on environmental and traffic conditions. Built with a Random Forest Classifier trained on the US Accidents dataset (~7.7M records), achieving **81% accuracy**.

---

## Live Demo
🔗 [https://radhey2565-accident-risk-prediction-appapp-rykbze.streamlit.app/](https://radhey2565-accident-risk-prediction-appapp-rykbze.streamlit.app/)

---

## Features
- Real-time accident risk prediction (High / Low)
- Interactive Streamlit dashboard
- Trained on 7.7M+ real accident records
- Handles class imbalance with balanced weighting
- Lightweight and fast inference via serialized model
- Cloud deployed on Streamlit Community Cloud

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core Programming |
| Streamlit | Web Application |
| Scikit-Learn | Machine Learning |
| Pandas & NumPy | Data Processing |
| Joblib | Model Serialization |
| Git & GitHub | Version Control |

---

## Project Structure

Accident_Risk_Prediction/

│

├── app/

│   └── app.py                   # Streamlit web application

│

├── models/

│   └── risk_model.pkl           # Trained Random Forest model

│

├── notebooks/

│   └── accident_analysis.ipynb  # EDA & model training

│

├── requirements.txt

└── README.md

---

## How It Works

1. User inputs driving conditions via the dashboard
2. Inputs are passed to the trained Random Forest model
3. Model predicts risk level based on learned patterns

**Input Features:**
- Temperature, Humidity, Pressure
- Visibility & Wind Speed
- Location (Lat/Lng)
- Time features (Hour, Month, Day)

**Output:**
- 🔴 High Risk
- 🟢 Low Risk

---

## Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | **81%** |
| Precision (Low Risk) | 91% |
| Precision (High Risk) | 74% |
| Recall (Low Risk) | 73% |
| Recall (High Risk) | 91% |
| F1-Score | 81% |
| Dataset | US Accidents (7.7M rows) |
| Algorithm | Random Forest Classifier |
| Class Handling | Balanced Weighting |

---

## Dataset

Download from Kaggle: [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

Place it in the project root as `accidents.csv` (not included in repo due to size).

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Radhey2565/accident-risk-prediction.git
cd accident-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## Future Improvements
- [ ] Improve accuracy further with XGBoost/LightGBM
- [ ] Add accident probability score (0-100%)
- [ ] Weather API integration for live conditions
- [ ] Accident hotspot mapping with Folium
- [ ] Prediction history tracking
- [ ] User authentication

---

## Author

**Radhey Mohan Singh**
Passionate about Machine Learning, Data Science, and AI-powered solutions.

[![GitHub](https://img.shields.io/badge/GitHub-Radhey2565-black?logo=github)](https://github.com/Radhey2565)

---

⭐ If you found this project helpful, consider giving it a star!

