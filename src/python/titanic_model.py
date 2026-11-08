"""Titanic survival prediction with logistic regression (Python version)."""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Inside Docker DATA_DIR=/app/data; locally it defaults to src/data
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
MAIN_TITLES = ["Mr", "Mrs", "Miss", "Master"]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Put the Kaggle file '{name}' in src/data/ (see README).")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_title(names):
    titles = names.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles.where(titles.isin(MAIN_TITLES), "Rare")


def learn_fill_values(df):
    """Fill values are learned from the training split only."""
    return {
        "Age": float(df["Age"].median()),
        "Fare": float(df["Fare"].median()),
        "Embarked": df["Embarked"].mode()[0],
    }


def prepare(df, fill, label):
    df = df.copy()
    print(f"\n[{label}] Missing values before cleaning:")
    print(df[["Age", "Fare", "Embarked", "Cabin"]].isna().sum().to_string())

    for col, value in fill.items():
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(value)
        print(f"[{label}] Filled {n_missing} missing '{col}' values with {value}")

    print(f"[{label}] Dropping columns not used as features: Cabin, Ticket")
    df = df.drop(columns=["Cabin", "Ticket"])

    df["Title"] = extract_title(df["Name"])
    print(f"[{label}] New feature 'Title' extracted from Name:")
    print(df["Title"].value_counts().to_string())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[{label}] New features 'FamilySize' (SibSp + Parch + 1) and 'IsAlone'; "
          f"{df['IsAlone'].sum()} passengers travelled alone")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print(f"[{label}] Encoded Sex as 0 = male, 1 = female")

    df = pd.get_dummies(df, columns=["Embarked", "Title"], dtype=int)
    print(f"[{label}] One-hot encoded Embarked and Title")
    return df


def feature_matrix(df, columns=None):
    base = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone"]
    dummies = [c for c in df.columns if c.startswith(("Embarked_", "Title_"))]
    X = df[base + sorted(dummies)]
    if columns is not None:  # make validation/test columns match training columns
        X = X.reindex(columns=columns, fill_value=0)
    return X


def main():
    section("STEP 1: Load training data")
    train_raw = load_csv("train.csv")
    print(train_raw.head().to_string())
    print(f"\nOverall survival rate in train.csv: {train_raw['Survived'].mean():.3f}")

    section("STEP 2: Explore the data")
    print("Survival rate by Sex:")
    print(train_raw.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train_raw.groupby("Pclass")["Survived"].mean().round(3).to_string())

    section("STEP 3: Split train.csv into training (80%) and validation (20%)")
    tr_raw, va_raw = train_test_split(
        train_raw, test_size=0.2, random_state=42, stratify=train_raw["Survived"]
    )
    print(f"Training rows: {len(tr_raw)}, validation rows: {len(va_raw)}")

    section("STEP 4: Clean and engineer features")
    fill = learn_fill_values(tr_raw)
    print(f"Fill values learned from training split: {fill}")
    tr = prepare(tr_raw, fill, "train")
    va = prepare(va_raw, fill, "validation")

    X_tr = feature_matrix(tr)
    X_va = feature_matrix(va, X_tr.columns)
    print(f"\nFeatures used ({X_tr.shape[1]}): {list(X_tr.columns)}")

    section("STEP 5: Fit logistic regression")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, tr["Survived"])
    coefs = pd.Series(model.coef_[0], index=X_tr.columns).sort_values()
    print("Model coefficients:")
    print(coefs.round(3).to_string())

    section("STEP 6: Accuracy")
    train_acc = accuracy_score(tr["Survived"], model.predict(X_tr))
    val_acc = accuracy_score(va["Survived"], model.predict(X_va))
    print(f"Training accuracy:   {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    section("STEP 7: Predict on test.csv")
    test_raw = load_csv("test.csv")
    te = prepare(test_raw, fill, "test")
    X_te = feature_matrix(te, X_tr.columns)
    preds = model.predict(X_te)
    out = pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": preds})
    out_path = DATA_DIR / "predictions_python.csv"
    out.to_csv(out_path, index=False)
    print(f"Predicted survival rate on test set: {preds.mean():.3f}")
    print(f"Saved {len(out)} predictions to {out_path}")

    section("STEP 8: Test-set accuracy")
    print("test.csv has no 'Survived' column, so true test accuracy is only available")
    print("by submitting predictions_python.csv to Kaggle.")
    gs_path = DATA_DIR / "gender_submission.csv"
    if gs_path.exists():
        gs = pd.read_csv(gs_path)
        merged = out.merge(gs, on="PassengerId", suffixes=("_pred", "_ref"))
        agree = (merged["Survived_pred"] == merged["Survived_ref"]).mean()
        print(f"Accuracy against gender_submission.csv (Kaggle's sample labels): {agree:.4f}")
    else:
        print("gender_submission.csv not found; skipping the comparison.")

    print("\nDone.")


if __name__ == "__main__":
    main()