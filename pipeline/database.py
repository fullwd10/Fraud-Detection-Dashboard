import pandas as pd
import sqlite3

def connect_to_db(db_name:str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name)
    print(f"Connection to {db_name} successful")
    return conn

def create_training_data_table(connection:sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS processed_training_data (
        transaction_id integer primary key autoincrement,
        type text,
        amount real, 
        oldBalanceOrig real,
        newBalanceOrig real, 
        oldBalanceDest real, 
        newBalanceDest real,
        oldBalanceOrigIsZero integer,
        oldBalanceDestIsZero integer,
        newBalanceDestIsZero integer,
        errorBalanceOrig real,
        errorBalanceDest real,
        isFraud integer)

        """)
    connection.commit()

def create_test_data_table(connection:sqlite3.Connection) -> str:
    cursor = connection.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS processed_test_data (
        transaction_id integer primary key autoincrement,
        type text,
        amount real, 
        oldBalanceOrig real,
        newBalanceOrig real, 
        oldBalanceDest real, 
        newBalanceDest real,
        oldBalanceOrigIsZero integer,
        oldBalanceDestIsZero integer,
        newBalanceDestIsZero integer,
        errorBalanceOrig real,
        errorBalanceDest real,
        fraudScore integer,
        encodedFeatures text,
        shapValues text,
        userIsFraud text,
        userComment text,
        isProcessed integer default 0)

        """)
    connection.commit()

def create_db_tables(connection:sqlite3.Connection) -> None:
    create_training_data_table(connection)
    create_test_data_table(connection)

VALID_TABLE_NAMES = {"processed_training_data", "processed_test_data"}

def get_data_from_table(connection:sqlite3.Connection, table_name:str) -> tuple:
    if table_name not in VALID_TABLE_NAMES:
        raise ValueError(f"Invalid table name: {table_name}")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * from {table_name}")
    row = cursor.fetchone()
    return row

def save_data_to_table(connection:sqlite3.Connection, table_name:str, df:pd.DataFrame) -> None:
    row = get_data_from_table(connection, table_name)
    if row is not None:
        print(f"Data already saved to {table_name}")
    else:
        df.to_sql(name=table_name, con=connection, if_exists='append', index=False)
        print(f"Data successfully saved to {table_name}")


















