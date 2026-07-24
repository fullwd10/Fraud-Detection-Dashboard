import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from .database import save_data_to_table, create_test_data_table, create_training_data_table
import sqlite3

RANDOM_STATE = 0

def get_data_from_csv(file_name:str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'oldbalanceDest':'oldBalanceDest', \
                'newbalanceDest':'newBalanceDest', 'newbalanceOrig':'newBalanceOrig'})
    print(f"Dataset of shape {df.shape} imported successfully")
    return df
    

def clean_data(data:pd.DataFrame) -> pd.DataFrame:
    data = data[(data['type'] == 'TRANSFER') | (data['type'] == 'CASH_OUT')] 
    data = data[data['nameDest'].str[0] != 'M']
    data = data.drop(['step','nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    return data

def feature_engineering(data:pd.DataFrame) -> pd.DataFrame:
    data['oldBalanceOrigIsZero'] = data['oldBalanceOrig'] == 0.0
    data['oldBalanceDestIsZero'] = data['oldBalanceDest'] == 0.0
    data['newBalanceDestIsZero'] = data['newBalanceDest'] == 0.0
    data['errorBalanceOrig'] = data['oldBalanceOrig'] - data['newBalanceOrig'] - data['amount']
    data['errorBalanceDest'] = data['oldBalanceDest'] - data['newBalanceDest'] - data['amount']
    return data

def split_data(data:pd.DataFrame) -> dict:
    y = data["isFraud"]
    X = data.drop("isFraud", axis=1)

    X_temp, X_test_raw, y_temp, y_test = train_test_split(X, y, train_size=0.8, stratify=y, 
                                                          random_state=RANDOM_STATE)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_temp, y_temp, train_size=0.75, stratify=y_temp,
                                                              random_state=RANDOM_STATE)

    X_train = pd.get_dummies(X_train_raw, drop_first=True)
    X_val = pd.get_dummies(X_val_raw, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, drop_first=True)

    return {"X_train":X_train, "y_train":y_train, "X_val":X_val, "y_val":y_val,
             "X_test":X_test, "y_test":y_test, "X_train_raw":X_train_raw, "X_test_raw":X_test_raw}

def data_preprocessing(file_name:str) -> dict:
    df = get_data_from_csv(file_name)
    df = clean_data(df)
    df = feature_engineering(df)
    df_dict = split_data(df)
    return df_dict

def process_training_data(connection:sqlite3.Connection, X_train_raw:pd.DataFrame, y_train:pd.DataFrame) -> None:
    create_training_data_table(connection)
    df_train = X_train_raw # Saves training data to DB without one-hot encoding
    df_train["isFraud"] = y_train
    save_data_to_table(connection, "processed_training_data", df_train)

def process_test_data(connection:sqlite3.Connection, X_test:pd.DataFrame, X_test_raw:pd.DataFrame, model:XGBClassifier) -> None:
    create_test_data_table(connection)
    df_test = X_test_raw
    xgb_probs = model.predict_proba(X_test)[:,1]
    df_test["fraudScore"] = 100*xgb_probs.round(1)
    df_test["encodedFeatures"] = X_test.apply(lambda row: row.to_json(), axis=1)
    save_data_to_table(connection, "processed_test_data", df_test)


