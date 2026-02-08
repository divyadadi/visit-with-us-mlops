import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

TARGET = "ProdTaken"

def main():
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    model = joblib.load("artifacts/model.joblib")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Metrics saved to artifacts/metrics.json")
    print(metrics)

if __name__ == "__main__":
    main()
