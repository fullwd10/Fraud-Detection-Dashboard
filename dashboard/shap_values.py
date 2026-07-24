import json
import pandas as pd
import sqlite3
import shap
import streamlit as st
from xgboost import XGBClassifier

@st.cache_resource
def get_trained_model():
    model = XGBClassifier()
    model.load_model("models/xgb_model.json")
    return model

@st.cache_resource
def get_shap_explainer() -> shap.TreeExplainer:
    return shap.TreeExplainer(get_trained_model())

def compute_shap_values(row:sqlite3.Row) -> shap.Explanation:
    df_row = pd.DataFrame([json.loads(row["encodedFeatures"])])
    xgb_explainer = get_shap_explainer()
    return xgb_explainer(df_row)

def format_shap_values_text(row_shap_values:shap.Explanation) -> str:
    base_value = row_shap_values.base_values[0]
    shap_sum = row_shap_values.values.reshape(-1).sum()
    top_features = sorted(zip(row_shap_values.feature_names, row_shap_values.values.reshape(-1)), key=lambda x:abs(x[1]), reverse=True)[0:5]

    shapValues_text = f"Base value (log-odds): {base_value:.3f} \n"
    shapValues_text += f"Sum of SHAP values: {shap_sum:.3f} \n\n"
    shapValues_text += "Top features:\n"
    for feature, value in top_features:
        direction = "↑ fraud risk" if value > 0 else "↓ fraud risk"
        shapValues_text += f"{feature}: {value:.3f} ({direction}) \n"
    return shapValues_text
