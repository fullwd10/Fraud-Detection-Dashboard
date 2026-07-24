from .database import connect_to_db, get_data_from_table, create_db_tables
from .modelling import train_model
from .processing import data_preprocessing, process_training_data, process_test_data

DB_NAME = "db/fraud_detection.db"
FILE_NAME = "input/PS_20174392719_1491204439457_log.csv"
TEST_DATA_NAME = "processed_test_data"

def run_pipeline(db_name:str, file_name:str) -> None:
    connection = connect_to_db(db_name)
    create_db_tables(connection)
    if get_data_from_table(connection, TEST_DATA_NAME) is not None:
        print("All data already processed and added to database")
        return
    df_dict = data_preprocessing(file_name)
    process_training_data(connection, df_dict["X_train_raw"], df_dict["y_train"])
    xgb_model = train_model(df_dict["X_train"], df_dict["y_train"], df_dict["X_val"], df_dict["y_val"])
    process_test_data(connection, df_dict["X_test"], df_dict["X_test_raw"], xgb_model)
    print("Data processing pipeline complete")


if __name__ == "__main__":
    run_pipeline(DB_NAME, FILE_NAME)