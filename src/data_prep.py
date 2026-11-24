"""Prétraitement commun aux deux modèles (split identique + scaling).

Usage : python -m src.data_prep
Sortie : data/processed/telco_splits.joblib
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
OUT_DIR  = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])

    # TotalCharges contient 11 espaces -> NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nan = df["TotalCharges"].isna().sum()
    print(f"TotalCharges : {n_nan} NaN supprimés")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Cible en 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def main():
    df = load_and_clean(RAW_PATH)
    print(f"Dataset nettoyé : {df.shape}, taux de churn = {df['Churn'].mean():.2%}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # One-hot encoding (identique pour les 2 étudiants)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    feature_names = X.columns.tolist()

    # Split stratifié : 60 / 20 / 20
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
    )

    # Scaling fit sur train uniquement
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(
        {
            "X_train": X_train_s, "y_train": y_train.values,
            "X_val":   X_val_s,   "y_val":   y_val.values,
            "X_test":  X_test_s,  "y_test":  y_test.values,
            "scaler": scaler, "feature_names": feature_names,
        },
        OUT_DIR / "telco_splits.joblib",
    )
    print(
        f"OK — train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
        f"features={len(feature_names)}"
    )


if __name__ == "__main__":
    main()