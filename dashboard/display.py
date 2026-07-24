import pandas as pd
import streamlit as st
import sqlite3

from database import get_transaction, get_transactions_count, update_userIsFraud, update_userComment, update_isProcessed

def load_offset_value() -> int:
    if "offset" not in st.session_state:
        st.session_state["offset"] = 0
    return st.session_state["offset"]

def display_title() -> None:
    st.title("Fraud Detection Dashboard", text_alignment="center")

def display_navigation_bar(connection:sqlite3.Connection) -> None:
    total = get_transactions_count(connection)
    max_val = max(total, 1)
    spacer, nav_col, label_col = st.columns([0.5,3,1.5]) #Correct ratios to display +/- buttons
    with nav_col:
        val = st.number_input(label="position", min_value=1, max_value=max_val, step=1)
        st.session_state["offset"] = val - 1
    with label_col:
        st.write("")
        st.write("")
        st.write(f"of {total}")

def display_fraud_score(row:sqlite3.Row) -> None:
    fraudScore = row["fraudScore"]
    st.metric(label="Fraud Score", value=fraudScore, border=True)

def display_score_nav_bar(connection:sqlite3.Connection) -> sqlite3.Row:
    score_col, spacer, nav_col = st.columns(spec=3)
    with nav_col:
        display_navigation_bar(connection)
    offset = load_offset_value()
    row = get_transaction(offset, connection)
    with score_col:
        display_fraud_score(row)
    return row

def display_features(row:sqlite3.Row) -> None:
    cols_to_display = ["type", "amount", "oldBalanceOrig", "newBalanceOrig",
                       "oldBalanceDest", "newBalanceDest"]
    col_labels = ["Type", "Amount", "Old bal. orig", "New bal. orig",
                 "Old bal. dest", "New bal. dest"]
    display_row = {col_labels[i]:row[cols_to_display[i]] for i in range(len(col_labels))}
    df_row = pd.DataFrame([display_row])
    st.caption(f"Transaction {row['transaction_id']}")
    st.dataframe(df_row, hide_index=True)

def display_conclusion(row:sqlite3.Row) -> str:
    userIsFraud = row["userIsFraud"]
    options = ["Fraud", "Not Fraud", "Secondary Review Required"]
    if userIsFraud in options:
        defaultindex = options.index(userIsFraud)
    else:
        defaultindex = None
    conclusion = st.selectbox("Conclusion", options, index=defaultindex, key=f"conclusion_{row['transaction_id']}")
    return conclusion

def display_comment(row:sqlite3.Row) -> str:
    userComment = row["userComment"]
    comment = st.text_area(
    "Comment",
    value=userComment if userComment is not None else "",
    placeholder="Add any notes on this transaction",
    key=f"comment_{row['transaction_id']}")
    return comment

def other_display_shap_values(row:sqlite3.Row) -> None:
    with st.container(border=True):
        st.caption("SHAP values")
        st.text(row["shapValues"])

def display_comment_shap(row:sqlite3.Row) -> str:
    left, right = st.columns(2)
    with left:
        other_display_shap_values(row)
    with right:
        comment = display_comment(row)
    return comment

def display_save_changes(connection:sqlite3.Connection, transaction_id:int, conclusion:str, comment:str) -> None:
    save_clicked = st.form_submit_button("Save")
    if save_clicked:
        try:
            update_userIsFraud(connection, transaction_id, conclusion)
            update_userComment(connection, transaction_id, comment)
            st.success("Changes saved successfully")
        except Exception:
            st.warning("Unable to save changes, please try again")

def display_confirm_next(connection:sqlite3.Connection, transaction_id:int, conclusion:str, comment:str) -> None:
    confirm_clicked = st.form_submit_button("Confirm & Next")
    if confirm_clicked:
        if conclusion is None:
            st.warning("Please select a conclusion before proceeding!")
        else:
            try:
                update_userIsFraud(connection, transaction_id, conclusion)
                update_userComment(connection, transaction_id, comment)
                update_isProcessed(connection, transaction_id)
                st.toast("Transaction processed successfully")
            except Exception:
                st.warning("Unable to process transaction, please try again")

def display_save_confirm(connection:sqlite3.Connection, transaction_id:int, conclusion:str, comment:str) -> None:
    with st.form("key"):
        display_save_changes(connection, transaction_id, conclusion, comment)
        display_confirm_next(connection, transaction_id, conclusion, comment)
