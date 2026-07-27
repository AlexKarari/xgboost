
"""
Gradient Boosting vs Random Forest on Telco Customer Churn.
"""
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
 
from src.gradient_boosting import GradientBoostingScratch, GBMConfig
 
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load + preprocess (reuse pipeline on the Telco dataset)
# ---------------------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
 
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
 
y = (df["Churn"] == "Yes").astype(int).values
 
cat_cols = df.select_dtypes(include="object").columns.drop(["customerID", "Churn"])
df_encoded = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), columns=cat_cols, drop_first=True)
 
X = df_encoded.values.astype(float)
feature_names = df_encoded.columns.tolist()
 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
 
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. Train scratch GBM with early stopping on validation set
# ---------------------------------------------------------------
config = GBMConfig(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=20,
    early_stopping_rounds=15,
    subsample=0.8,
    random_state=RANDOM_STATE,
)
 
gbm_scratch = GradientBoostingScratch(config)
gbm_scratch.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)

# ---------------------------------------------------------------
# 3. Validate against sklearn GradientBoostingClassifier
# ---------------------------------------------------------------
gbm_sklearn = GradientBoostingClassifier(
    n_estimators=len(gbm_scratch.trees),
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=20,
    subsample=0.8,
    random_state=RANDOM_STATE,
)
gbm_sklearn.fit(X_train_s, y_train)

# ---------------------------------------------------------------
# 4. Compare against Random Forest
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=10,
    random_state=RANDOM_STATE, oob_score=True
)
rf.fit(X_train_s, y_train)
 
 
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")
 
 
results = {}
for name, model, is_scratch in [
    ("GBM (scratch)", gbm_scratch, True),
    ("GBM (sklearn)", gbm_sklearn, False),
    ("Random Forest (Day 8)", rf, False),
]:
    if is_scratch:
        proba = model.predict_proba(X_test_s)
        pred = model.predict(X_test_s)
    else:
        proba = model.predict_proba(X_test_s)[:, 1]
        pred = model.predict(X_test_s)
    evaluate(name, y_test, pred, proba)
    results[name] = {"accuracy": accuracy_score(y_test, pred),
                      "f1": f1_score(y_test, pred),
                      "roc_auc": roc_auc_score(y_test, proba)}
 
print("\nScratch vs sklearn implementation match check:")
scratch_proba = gbm_scratch.predict_proba(X_test_s)
sklearn_proba = gbm_sklearn.predict_proba(X_test_s)[:, 1]
print(f"Max |proba diff|: {np.max(np.abs(scratch_proba - sklearn_proba)):.4f}")