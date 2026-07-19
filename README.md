#  Used Phone Price Prediction

##  Project Overview

This project predicts the resale price of used smartphones using Machine Learning. It is built using Python and Scikit-learn and demonstrates the complete ML workflow, from data preprocessing to model deployment.

> **Note:** The dataset used in this project is **synthetically generated** and was obtained from Kaggle. It is designed to simulate real-world used smartphone resale data for learning and experimentation.

---

##  Objective

To build a machine learning model that accurately predicts the resale price of a used smartphone based on its specifications.

---

##  Dataset

- **Source:** Kaggle
- **Rows:** ~1 Million
- **Type:** Synthetic dataset (designed to resemble real-world resale market data)

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

##  Machine Learning Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Stratified Train-Test Split
- Feature Engineering
- Data Preprocessing using ColumnTransformer
- One-Hot Encoding
- Feature Scaling using StandardScaler
- Model Training
- Model Evaluation
- Model Serialization using Joblib
- Prediction on New Data

---

##  Models Used

- Linear Regression
- Random Forest Regressor

The Random Forest Regressor was selected as the final model because it achieved the best performance.

---

##  Model Performance

| Metric | Value |
|---------|--------|
| RMSE | 3063 |
| R^2 Score | 0.968 |

---

##  Project Structure

```
Used-Phone-Price-Prediction/
│
├── data/
│   ├── used_phone_price.csv
│   ├── input_mobile_resale_price.csv
│   └── output_mobile_resale_price.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── main.py
├── model_selection.py
├── requirements.txt
└── README.md
```

---

##  How to Run

1. Clone the repository.

2. Install the required libraries.

```
pip install -r requirements.txt
```

3. Run the prediction script.

```
python main.py
```

The predicted resale prices will be saved in:

```
output_mobile_resale_price.csv
```

---

##  Future Improvements

- Hyperparameter tuning
- Model deployment using Flask or FastAPI
- Interactive web application
- Support for additional smartphone brands and features

---

##  Author

**Uttaran Kanungo**
