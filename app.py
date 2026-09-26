import os
import json
import pickle
import traceback
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# Initialize Flask application
app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Loan_default.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
DEFAULT_MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "default_model.json")

# Model Registry Configuration
MODEL_DEFINITIONS = {
    "Logistic Regression": {
        "filename": "logistic_regression_model.pkl",
        "description": "Standard linear model using log-odds transformation. Highly interpretable baseline."
    },
    "Decision Tree": {
        "filename": "decision_tree_model.pkl",
        "description": "Non-linear decision tree splitting features hierarchically to maximize information gain."
    },
    "Random Forest": {
        "filename": "random_forest_model.pkl",
        "description": "Ensemble of 100 decorrelated decision trees using bagging to minimize variance."
    },
    "AdaBoost": {
        "filename": "adaboost_model.pkl",
        "description": "Adaptive boosting ensemble that iteratively focuses learning on harder, misclassified cases."
    },
    "Gradient Boosting": {
        "filename": "gradient_boosting_model.pkl",
        "description": "Sequentially fits trees to the negative gradient of the loss function. High predictive accuracy."
    }
}

# The 16 exact feature names expected by the trained pipeline
FEATURE_NAMES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner"
]

# Global In-Memory Caches
loaded_models = {}
model_statuses = {}
dataset_df = None
model_metrics = {}
active_default_model = "Random Forest"

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def load_default_model_setting():
    global active_default_model
    if os.path.exists(DEFAULT_MODEL_CONFIG_PATH):
        try:
            with open(DEFAULT_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                active_default_model = data.get("default_model", "Random Forest")
        except Exception as e:
            print(f"Notice reading default_model.json: {e}")

def save_default_model_setting(model_name):
    global active_default_model
    active_default_model = model_name
    try:
        with open(DEFAULT_MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"default_model": model_name}, f, indent=2)
    except Exception as e:
        print(f"Notice saving default_model.json: {e}")

def load_all_models():
    """Load all .pkl files from models/ directory once on server startup."""
    global loaded_models, model_statuses
    print("=" * 60)
    print("Initializing Loan Default Prediction ML Models...")
    
    for name, config in MODEL_DEFINITIONS.items():
        filename = config["filename"]
        model_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(model_path):
            print(f"[-] Model file not found: {filename}")
            model_statuses[name] = {
                "available": False,
                "filename": filename,
                "error": "File not found"
            }
            continue

        try:
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)
            loaded_models[name] = pipeline
            model_statuses[name] = {
                "available": True,
                "filename": filename,
                "type": type(pipeline).__name__
            }
            print(f"[+] Loaded {name} successfully from {filename}")
        except Exception as e:
            print(f"[x] Error loading {filename}: {e}")
            model_statuses[name] = {
                "available": False,
                "filename": filename,
                "error": str(e)
            }
    
    print(f"Loaded {len(loaded_models)} of {len(MODEL_DEFINITIONS)} models into memory.")
    print("=" * 60)

def load_dataset():
    """Load original CSV into pandas DataFrame for fast pagination and search."""
    global dataset_df
    if dataset_df is None and os.path.exists(DATASET_PATH):
        try:
            print(f"Caching dataset from {DATASET_PATH}...")
            dataset_df = pd.read_csv(DATASET_PATH)
            print(f"[+] Cached {len(dataset_df):,} records ({dataset_df.shape[1]} columns).")
        except Exception as e:
            print(f"[-] Failed to load dataset: {e}")

def load_metrics():
    """Load evaluated model metrics from model_metrics.json."""
    global model_metrics
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
            print(f"[+] Loaded metrics for {len(model_metrics)} models.")
        except Exception as e:
            print(f"Notice reading model_metrics.json: {e}")

# Startup Initialization
load_default_model_setting()
load_all_models()
load_metrics()
load_dataset()


# ==============================================================================
# Web Routes
# ==============================================================================

@app.route("/")
def home():
    """Homepage route with summary cards and workflow introduction."""
    total_rows = len(dataset_df) if dataset_df is not None else 255347
    return render_template(
        "index.html",
        active_page="home",
        default_model=active_default_model,
        available_models_count=len(loaded_models),
        total_rows=total_rows,
        models=model_statuses
    )

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Prediction page route (GET) and prediction handling (POST)."""
    if request.method == "POST":
        return handle_prediction_request()
    
    return render_template(
        "predict.html",
        active_page="predict",
        default_model=active_default_model,
        models=model_statuses
    )

def handle_prediction_request():
    """Process prediction form submission with real trained ML pipeline."""
    try:
        # Support both JSON payload and Form-encoded data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        selected_model_name = data.get("model_name", active_default_model).strip()
        
        # Validate model availability
        if selected_model_name not in loaded_models:
            return jsonify({
                "success": False,
                "error": f"Selected model '{selected_model_name}' is not available or not loaded."
            }), 400

        model = loaded_models[selected_model_name]

        # Extract and validate inputs
        input_dict = {}
        for feature in FEATURE_NAMES:
            if feature not in data or data[feature] == "":
                return jsonify({
                    "success": False,
                    "error": f"Missing required input parameter: '{feature}'."
                }), 400

        # Type conversion and validation
        try:
            input_dict["Age"] = int(data["Age"])
            input_dict["Income"] = float(data["Income"])
            input_dict["LoanAmount"] = float(data["LoanAmount"])
            input_dict["CreditScore"] = int(data["CreditScore"])
            input_dict["MonthsEmployed"] = int(data["MonthsEmployed"])
            input_dict["NumCreditLines"] = int(data["NumCreditLines"])
            input_dict["InterestRate"] = float(data["InterestRate"])
            input_dict["LoanTerm"] = int(data["LoanTerm"])
            input_dict["DTIRatio"] = float(data["DTIRatio"])

            input_dict["Education"] = str(data["Education"])
            input_dict["EmploymentType"] = str(data["EmploymentType"])
            input_dict["MaritalStatus"] = str(data["MaritalStatus"])
            input_dict["HasMortgage"] = str(data["HasMortgage"])
            input_dict["HasDependents"] = str(data["HasDependents"])
            input_dict["LoanPurpose"] = str(data["LoanPurpose"])
            input_dict["HasCoSigner"] = str(data["HasCoSigner"])
        except ValueError as ve:
            return jsonify({
                "success": False,
                "error": f"Data type conversion failed: {str(ve)}"
            }), 400

        # Construct single-row DataFrame with exact feature order
        input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]

        # Execute ML prediction
        prediction_val = int(model.predict(input_df)[0])
        
        # Calculate probabilities if supported
        default_probability = 0.0
        no_default_probability = 0.0
        confidence_probability = 0.0

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            no_default_probability = float(proba[0] * 100.0)
            default_probability = float(proba[1] * 100.0)
            confidence_probability = max(no_default_probability, default_probability)
        else:
            confidence_probability = 100.0 if prediction_val == 1 else 0.0
            default_probability = 100.0 if prediction_val == 1 else 0.0

        # Risk categorization based on estimated default probability
        if default_probability < 25.0:
            risk_level = "Low Risk"
        elif default_probability < 50.0:
            risk_level = "Moderate Risk"
        elif default_probability < 75.0:
            risk_level = "High Risk"
        else:
            risk_level = "Critical Risk"

        prediction_label = "Loan Default" if prediction_val == 1 else "No Loan Default"

        return jsonify({
            "success": True,
            "prediction": prediction_val,
            "prediction_label": prediction_label,
            "model_name": selected_model_name,
            "default_probability": round(default_probability, 2),
            "no_default_probability": round(no_default_probability, 2),
            "confidence_probability": round(confidence_probability, 2),
            "risk_level": risk_level
        })

    except Exception as e:
        print(f"Prediction Error: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"An error occurred while generating prediction: {str(e)}"
        }), 500

@app.route("/comparison")
def comparison_page():
    """Model comparison and benchmark page."""
    return render_template(
        "comparison.html",
        active_page="comparison",
        default_model=active_default_model,
        metrics=model_metrics,
        models=model_statuses
    )

@app.route("/dataset")
def dataset_page():
    """Dataset explorer page."""
    total_rows = len(dataset_df) if dataset_df is not None else 255347
    total_cols = dataset_df.shape[1] if dataset_df is not None else 18
    return render_template(
        "dataset.html",
        active_page="dataset",
        default_model=active_default_model,
        total_rows=total_rows,
        total_cols=total_cols
    )

@app.route("/about")
def about_page():
    """About project page."""
    return render_template(
        "about.html",
        active_page="about",
        default_model=active_default_model,
        models=model_statuses
    )

@app.route("/download-dataset")
def download_dataset():
    """Stream download original dataset CSV."""
    if os.path.exists(DATASET_PATH):
        return send_file(
            DATASET_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="Loan_default.csv"
        )
    return jsonify({"error": "Dataset file not found on server."}), 404


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route("/api/models", methods=["GET"])
def api_models():
    """Return available models status."""
    return jsonify({
        "success": True,
        "default_model": active_default_model,
        "models": model_statuses,
        "metrics": model_metrics
    })

@app.route("/api/default-model", methods=["GET", "POST"])
def api_default_model():
    """Get or update active default model."""
    if request.method == "POST":
        data = request.get_json() or {}
        new_model = data.get("default_model", "").strip()
        if new_model in MODEL_DEFINITIONS:
            save_default_model_setting(new_model)
            return jsonify({
                "success": True,
                "message": f"Default model set to {new_model}",
                "default_model": active_default_model
            })
        return jsonify({
            "success": False,
            "error": f"Invalid model name '{new_model}'"
        }), 400

    return jsonify({
        "success": True,
        "default_model": active_default_model
    })

@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    """Fast server-side pagination, search, and filtering on the dataset."""
    global dataset_df
    if dataset_df is None:
        load_dataset()

    if dataset_df is None:
        return jsonify({"success": False, "error": "Dataset unavailable"}), 500

    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(max(int(request.args.get("per_page", 50)), 10), 100)
        search_query = request.args.get("search", "").strip().lower()
        default_filter = request.args.get("default_filter", "all").strip()

        filtered_df = dataset_df

        # Filter by default status
        if default_filter in ["0", "1"]:
            filtered_df = filtered_df[filtered_df["Default"] == int(default_filter)]

        # Search across text and ID columns
        if search_query:
            mask = (
                filtered_df["LoanID"].astype(str).str.lower().str.contains(search_query, na=False) |
                filtered_df["Education"].astype(str).str.lower().str.contains(search_query, na=False) |
                filtered_df["EmploymentType"].astype(str).str.lower().str.contains(search_query, na=False) |
                filtered_df["LoanPurpose"].astype(str).str.lower().str.contains(search_query, na=False)
            )
            filtered_df = filtered_df[mask]

        total_rows = len(filtered_df)
        total_pages = max((total_rows + per_page - 1) // per_page, 1)

        # Slice pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        sliced_df = filtered_df.iloc[start_idx:end_idx]

        records = sliced_df.to_dict(orient="records")

        return jsonify({
            "success": True,
            "page": page,
            "per_page": per_page,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "rows": records
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==============================================================================
# Error Handlers
# ==============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return render_template(
        "index.html",
        active_page="home",
        default_model=active_default_model,
        available_models_count=len(loaded_models),
        total_rows=len(dataset_df) if dataset_df is not None else 255347,
        models=model_statuses
    ), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal Server Error occurred. Please try again."
    }), 500


if __name__ == "__main__":
    # Local development server entrypoint
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[+] Loan Default Prediction System running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
