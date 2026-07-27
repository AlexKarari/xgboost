
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
 