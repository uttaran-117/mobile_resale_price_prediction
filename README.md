#  Mobile Resale Price Prediction

A Machine Learning project that predicts the resale price of used smartphones based on their specifications, condition, and market-related features.

---

##  Project Overview

This project uses a machine learning pipeline to estimate the resale price of a used smartphone. The complete workflow includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Prediction on new data

---

##  Features

- Data preprocessing using Scikit-learn Pipelines
- Automatic handling of numerical and categorical features
- One-Hot Encoding for categorical variables
- Feature Scaling
- Random Forest Regressor for prediction
- Predict resale price for new smartphone data

---

##  Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
mobile_resale_price_prediction/
│
├── data/
│   ├── input_mobile_resale_price.csv
│   └── output_mobile_resale_price.csv
│
├── models/
│   └── pipeline.pkl
│
├── main.py
├── model_selection.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Model Performance

| Metric | Score |
|--------|--------|
| R² Score | **0.968** |
| RMSE | **≈ 3063** |

---

## ## Dataset

The original dataset has been excluded from this repository.

Download it from Kaggle:

[[Dataset Link](https://www.kaggle.com/datasets/sharmajicoder/used-phone-price-prediction-dataset)]

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/uttaran-117/mobile_resale_price_prediction.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run prediction

```bash
python main.py
```

Predictions will be saved as:

```
data/output_mobile_resale_price.csv
```

---

## ⚙️ Model Training

To retrain the model using the original dataset, run:

```bash
python model_selection.py
```

**Note:** The trained `model.pkl` file is intentionally excluded from this repository because of its large size.

---

##  Future Improvements

- Hyperparameter tuning
- XGBoost/LightGBM implementation
- Flask/FastAPI deployment
- Interactive web interface
- Model explainability using SHAP

---

##  Author

**Uttaran Kanungo**

GitHub: https://github.com/uttaran-117

---

##  If you found this project useful, consider giving it a star!
