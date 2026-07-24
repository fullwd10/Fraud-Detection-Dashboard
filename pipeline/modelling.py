from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import pandas as pd
import sqlite3
import shap
import json

RANDOM_STATE = 0

def train_model(X_train:pd.DataFrame, y_train:pd.DataFrame, X_val:pd.DataFrame, y_val:pd.DataFrame) -> XGBClassifier:
    xgb_model = XGBClassifier(learning_rate=0.05, n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    print('XGBoost model trained')
    auprc = average_precision_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
    print(f'AUPRC for XGBoost model on validation data: {auprc:.4f}')
    xgb_model.save_model("models/xgb_model.json")
    print("Model saved to file models/xgb_model.json")
    return xgb_model






