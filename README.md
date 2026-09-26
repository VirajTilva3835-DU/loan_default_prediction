# Loan Default Prediction - Machine Learning Web Platform

An end-to-end Machine Learning web application for credit risk assessment and Loan Default Prediction. The system features 5 trained scikit-learn classification pipelines, real-time prediction inference, dynamic model comparison with interactive charts, a public default model selection system, and a paginated dataset explorer for 255,000+ loan records.

---

## 1. Project Overview

Predicting loan default is a vital binary classification task in financial risk management. Given demographic, financial, and credit parameters of a loan applicant, the system predicts:
- **`0` (No Loan Default)**: Low credit risk; high likelihood of consistent repayment.
- **`1` (Loan Default)**: Elevated credit risk; probability of default or delinquency.

### Key Highlights
- **Real Trained ML Pipelines:** Connects directly to serialized `.pkl` pipelines (`StandardScaler` + `OneHotEncoder(drop='first')` + Classifier). No hard-coded heuristics or synthetic predictions.
- **5 Benchmarked Algorithms:** Logistic Regression, Decision Tree, Random Forest, AdaBoost, and Gradient Boosting.
- **Public Default Model Selector:** Users can set any model as the default. Persisted via browser `localStorage` and backend configuration.
- **Educational Pop-up Modal:** Clarifies that the platform is for educational demonstration and does not constitute real financial decisions.
- **Full Dataset Explorer:** Inspect all 255,347 training records via server-side pagination and real-time search with zero UI lag.
- **No Login / Authentication Required:** Open, public ML dashboard.

---

## 2. Folder Structure

```text
week 4/loan_default_prediction/
│
├── app.py                      # Main Flask backend application & API routes
├── requirements.txt            # Python dependencies (Flask, pandas, scikit-learn, etc.)
├── model_metrics.json          # Verified test set evaluation & 5-fold CV metrics
├── train_models.py             # Script used to fit pipelines and export .pkl models
├── HOW_TO_RUN.md               # Quick execution instructions
├── run.bat                     # 1-click Windows batch launcher
├── README.md                   # Complete technical documentation
│
├── dataset/
│   └── Loan_default.csv        # 255,347 loan training records (25 MB)
│
├── models/                     # Serialized scikit-learn Pipeline objects
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── adaboost_model.pkl
│   └── gradient_boosting_model.pkl
│
├── templates/                  # Jinja2 HTML Templates
│   ├── base.html               # Base layout, navbar & disclaimer modal
│   ├── index.html              # Homepage with hero & metrics cards
│   ├── predict.html            # Prediction input form & real-time result gauge
│   ├── comparison.html         # Model comparison table & Chart.js visualizations
│   ├── dataset.html            # Server-side pagination & CSV dataset explorer
│   └── about.html              # Machine learning architecture & feature glossary
│
└── static/
    ├── css/
    │   └── style.css           # Modern dark-mode dashboard styling
    ├── js/
    │   └── script.js           # AJAX form handlers, default model sync, pagination
    └── images/
```

---

## 3. How `.pkl` Models Are Used

Each `.pkl` file encapsulates a full `sklearn.pipeline.Pipeline` object containing:
1. **Feature Preprocessor (`ColumnTransformer`):**
   - **Numerical Features (9):** `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio` normalized via `StandardScaler()`.
   - **Categorical Features (7):** `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner` transformed via `OneHotEncoder(drop='first', handle_unknown='ignore')`.
2. **Estimator:** The trained classification model (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, or `GradientBoostingClassifier`).

Because preprocessing is serialized inside the pipeline, the backend simply passes a raw pandas DataFrame matching the original column names:
```python
import pickle
import pandas as pd

with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Raw input DataFrame
prediction = model.predict(input_df)       # Returns [0] or [1]
probability = model.predict_proba(input_df) # Returns [[P(No Default), P(Default)]]
```
Models are loaded into memory **once** at server startup in a dictionary:
```python
models = {
    "Logistic Regression": loaded_logistic_model,
    "Decision Tree": loaded_decision_tree_model,
    "Random Forest": loaded_random_forest_model,
    "AdaBoost": loaded_adaboost_model,
    "Gradient Boosting": loaded_gradient_boosting_model
}
```

---

## 4. How to Install Dependencies

Make sure you have Python 3.10+ installed.

Run:
```bash
pip install -r requirements.txt
```

---

## 5. How to Run the Website

Run:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

*(On Windows, you can also simply double-click `run.bat`)*

---

## 6. How to Add or Replace Models

To add or update a model:
1. Train your scikit-learn pipeline on the 16 features:
   ```python
   from sklearn.pipeline import Pipeline
   pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', your_model)])
   pipeline.fit(X_train, y_train)
   ```
2. Save it into the `models/` folder as a `.pkl` file (e.g. `models/random_forest_model.pkl`):
   ```python
   import pickle
   with open("models/random_forest_model.pkl", "wb") as f:
       pickle.dump(pipeline, f)
   ```
3. Restart the Flask app (`python app.py`). The backend automatically inspects the `models/` directory and loads the new pipeline.

---

## 7. How to Change the Default Model

1. Navigate to the **Model Comparison** page.
2. In the **Default Prediction Model Selection** card, select the radio button for your desired algorithm.
3. Click **"Save as Default Model"** (or click the "Set as Default" button next to any model in the table).
4. The selection is stored in browser `localStorage` and synchronized with the backend.
5. The selected model will automatically be pre-selected every time the **Predict** page is opened.

---

## 8. How Model Comparison Works

The **Model Comparison** page benchmarks all 5 algorithms using verified evaluation scores computed from a 20% hold-out test set (51,070 records) and 5-fold cross-validation:

| Model | Accuracy | Precision | Recall | F1 Score | 5-Fold CV Mean F1 | 5-Fold CV Std F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **88.64%** | 63.66% | **5.11%** | **0.0946** | 0.0919 | ±0.0027 | Loaded |
| **Random Forest** | 88.44% | **83.78%** | 0.52% | 0.0104 | 0.0109 | ±0.0025 | Loaded |
| **AdaBoost** | 88.54% | 64.73% | 2.82% | 0.0540 | 0.0543 | ±0.0025 | Loaded |
| **Decision Tree** | 88.18% | 43.29% | 5.82% | 0.1026 | 0.1117 | ±0.0099 | Loaded |
| **Logistic Regression**| 88.52% | 60.31% | 3.30% | 0.0627 | 0.0642 | ±0.0027 | Loaded |

### Visual Comparisons
- **Accuracy, Precision & Recall Bar Chart:** Compares false-positive avoidance (precision) vs default capture rate (recall).
- **F1 Score & Cross-Validation Chart:** Illustrates model stability across stratified folds.

---

## 9. Educational Disclaimer

> **Important Notice:** This system is an educational machine learning demonstration. The predictions are statistical estimations produced by machine learning models and are NOT actual financial or credit lending decisions. This system must not be used to approve or reject real-world loans.
