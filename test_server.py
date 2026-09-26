import urllib.request
import json

def test_url(url):
    req = urllib.request.urlopen(url)
    print(f"{url} -> Status: {req.status}")
    return req.read().decode("utf-8")

print("--- Testing GET routes ---")
test_url("http://127.0.0.1:5000/")
test_url("http://127.0.0.1:5000/predict")
test_url("http://127.0.0.1:5000/comparison")
test_url("http://127.0.0.1:5000/dataset")
test_url("http://127.0.0.1:5000/about")

print("\n--- Testing API Models ---")
res = test_url("http://127.0.0.1:5000/api/models")
data = json.loads(res)
print("Models API Success:", data["success"], "Default:", data["default_model"])

print("\n--- Testing API Dataset Pagination ---")
res = test_url("http://127.0.0.1:5000/api/dataset?page=1&per_page=2")
d_data = json.loads(res)
print("Dataset Rows count:", len(d_data["rows"]), "Total rows:", d_data["total_rows"], "First row LoanID:", d_data["rows"][0]["LoanID"])

print("\n--- Testing Prediction POST API with all 5 models ---")
for model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting"]:
    sample_payload = {
        "model_name": model_name,
        "Age": 42, "Income": 85000, "LoanAmount": 28000, "CreditScore": 710,
        "MonthsEmployed": 52, "NumCreditLines": 2, "InterestRate": 6.8,
        "LoanTerm": 36, "DTIRatio": 0.26, "Education": "Bachelor's",
        "EmploymentType": "Full-time", "MaritalStatus": "Married",
        "HasMortgage": "No", "HasDependents": "Yes", "LoanPurpose": "Home",
        "HasCoSigner": "Yes"
    }
    req = urllib.request.Request(
        "http://127.0.0.1:5000/predict",
        data=json.dumps(sample_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    pred_data = json.loads(resp.read().decode("utf-8"))
    print(f"[{model_name}] -> Pred: {pred_data['prediction_label']}, Default Prob: {pred_data['default_probability']}%, Conf: {pred_data['confidence_probability']}%, Risk: {pred_data['risk_level']}")

print("\n--- Testing Changing Default Model API ---")
req = urllib.request.Request(
    "http://127.0.0.1:5000/api/default-model",
    data=json.dumps({"default_model": "Gradient Boosting"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
print("Set Default Model:", json.loads(resp.read().decode("utf-8")))

print("\nALL SERVER TESTS PASSED SUCCESSFULLY!")
