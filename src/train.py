import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

TARGET = "ProdTaken"

def main():
    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int)

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 4],
        "model__max_features": ["sqrt", "log2"]
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(grid.best_estimator_, "artifacts/model.joblib")

    with open("artifacts/best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=2)

    with open("artifacts/best_cv_score.json", "w") as f:
        json.dump({"best_cv_f1": float(grid.best_score_)}, f, indent=2)

    print("✅ Best model saved to artifacts/model.joblib")
    print("✅ Best params saved to artifacts/best_params.json")

if __name__ == "__main__":
    main()
