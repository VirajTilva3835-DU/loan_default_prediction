import os
import time
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def train_and_export():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "Loan_default.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    t0 = time.time()
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")

    X = df.drop(columns=["Default", "LoanID"])
    y = df["Default"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    print("Splitting dataset (80/20 stratify)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models_config = {
        "logistic_regression_model.pkl": {
            "name": "Logistic Regression",
            "classifier": LogisticRegression(max_iter=1000, random_state=42)
        },
        "decision_tree_model.pkl": {
            "name": "Decision Tree",
            "classifier": DecisionTreeClassifier(max_depth=10, random_state=42)
        },
        "random_forest_model.pkl": {
            "name": "Random Forest",
            "classifier": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        },
        "adaboost_model.pkl": {
            "name": "AdaBoost",
            "classifier": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
        },
        "gradient_boosting_model.pkl": {
            "name": "Gradient Boosting",
            "classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        }
    }

    metrics_output = {}

    # Verified 5-Fold Cross Validation values computed in Week 5.ipynb
    known_cv = {
        "Logistic Regression": {"mean_f1": 0.064208, "std_f1": 0.002691},
        "Decision Tree": {"mean_f1": 0.111687, "std_f1": 0.009876},
        "Random Forest": {"mean_f1": 0.010883, "std_f1": 0.002485},
        "AdaBoost": {"mean_f1": 0.054265, "std_f1": 0.002468},
        "Gradient Boosting": {"mean_f1": 0.091856, "std_f1": 0.002727}
    }

    for filename, config in models_config.items():
        name = config["name"]
        clf = config["classifier"]
        print(f"\n--- Training {name} ---")
        t_start = time.time()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ]
        )

        pipeline.fit(X_train, y_train)
        train_time = time.time() - t_start
        print(f"Trained in {train_time:.2f}s")

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cv_info = known_cv.get(name, {"mean_f1": 0.0, "std_f1": 0.0})

        metrics_output[name] = {
            "model_name": name,
            "filename": filename,
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1_score": round(float(f1), 6),
            "cv_mean_f1": round(float(cv_info["mean_f1"]), 6),
            "cv_std_f1": round(float(cv_info["std_f1"]), 6),
            "status": "Available",
            "train_time_sec": round(train_time, 2)
        }

        print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # Save model pipeline
        save_path = os.path.join(models_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved pipeline to: {save_path} ({os.path.getsize(save_path) / (1024*1024):.2f} MB)")

    # Save metrics JSON
    metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"\nAll models trained and saved successfully! Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_and_export()
