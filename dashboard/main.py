import sqlite3

from database import DB_NAME, connect_to_db, get_transaction, get_transaction_id, save_shap_values_to_db
from shap_values import compute_shap_values, format_shap_values_text
from display import (
    load_offset_value,
    display_title,
    display_score_nav_bar,
    display_features,
    display_comment_shap,
    display_conclusion,
    display_save_confirm,
)

def refresh_shap_values(connection:sqlite3.Connection, transaction_id:int, row:sqlite3.Row) -> sqlite3.Row:
    row_shap_values = compute_shap_values(row)
    save_shap_values_to_db(connection, transaction_id, format_shap_values_text(row_shap_values))
    return get_transaction(load_offset_value(), connection)

def display_transaction_review(row:sqlite3.Row) -> tuple[str, str]:
    display_features(row)
    comment = display_comment_shap(row)
    conclusion = display_conclusion(row)
    return comment, conclusion

def main():
    conn = connect_to_db(DB_NAME)
    display_title()
    row = display_score_nav_bar(conn)
    transaction_id = get_transaction_id(row)
    row = refresh_shap_values(conn, transaction_id, row)
    comment, conclusion = display_transaction_review(row)
    display_save_confirm(conn, transaction_id, conclusion, comment)

if __name__ == '__main__':
    main()
