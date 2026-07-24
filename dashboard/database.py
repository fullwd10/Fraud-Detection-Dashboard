import sqlite3
import streamlit as st

DB_NAME = "db/fraud_detection.db"

@st.cache_resource
def connect_to_db(db_name:str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name, check_same_thread=False)
    return conn

def get_transaction(offset:int, connection:sqlite3.Connection) -> sqlite3.Row:
    connection.row_factory = sqlite3.Row # modifies what fetchall returns - now return Row object instead of tuple
    cursor = connection.cursor()
    cursor.execute("""
    SELECT * FROM processed_test_data
    WHERE (fraudScore > 5 AND fraudScore < 95) AND (isProcessed = 0)
    ORDER BY fraudScore DESC
    LIMIT 1
    OFFSET ?
    """, (offset,))
    row = cursor.fetchone()
    connection.commit()
    return row

def get_transaction_id(row):
    return row["transaction_id"]

def get_transactions_count(connection:sqlite3.Connection) -> int:
    cursor = connection.cursor()
    cursor.execute("""
    SELECT COUNT(*) FROM processed_test_data
    WHERE (fraudScore > 5 AND fraudScore < 95) AND (isProcessed = 0)
    """)
    count = cursor.fetchone()
    connection.commit()
    return count[0]

def save_shap_values_to_db(connection:sqlite3.Connection, transaction_id:int, shap_values:str) -> None:
    cursor = connection.cursor()
    cursor.execute(""" 
    UPDATE processed_test_data
    SET shapValues = ?
    WHERE transaction_id = ?
    """, (shap_values, transaction_id))
    connection.commit()

def update_userIsFraud(connection:sqlite3.Connection, transaction_id:int, conclusion:str) -> None:
    cursor = connection.cursor()
    cursor.execute("""
    UPDATE processed_test_data
    SET userIsFraud = ?
    WHERE transaction_id = ?
    """, (conclusion, transaction_id))
    connection.commit()

def update_userComment(connection:sqlite3.Connection, transaction_id:int, comment:str) -> None:
    cursor = connection.cursor()
    cursor.execute("""
    UPDATE processed_test_data
    SET userComment = ?
    WHERE transaction_id = ?
    """, (comment, transaction_id))
    connection.commit()

def update_isProcessed(connection:sqlite3.Connection, transaction_id:int) -> None:
    cursor = connection.cursor()
    cursor.execute("""
    UPDATE processed_test_data
    SET isProcessed = 1
    WHERE transaction_id = ?
    """, (transaction_id,))
    connection.commit()
