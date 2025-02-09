import os
import time
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://myuser:mypassword@db:5432/training_data")
host = "localhost"  
port = "5432"
dbname = "loan_database"  
user = "myuser" 
password = "mypassword"  

def wait_for_database():
    
    max_retries = 10
    retries = 0
    while retries < max_retries:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")  
            cursor.close()
            conn.close()
            print("✅ Database is ready.")
            return
        except Exception as e:
            print(f"⚠️ Waiting for database... ({retries+1}/{max_retries})")
            time.sleep(5)
            retries += 1
    print("❌ Database connection failed after multiple attempts.")


def fetch_data(query="SELECT * FROM training_data;"):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()  
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        df = pd.DataFrame(result, columns=columns)
        print("✅ Data fetched successfully.")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def save_to_monitoring(df):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cursor:
            data_to_insert = df.values.tolist()
            column_list_str = ', '.join(df.columns)

            insert_query = f"""
            INSERT INTO monitoring ({column_list_str})
            VALUES %s
            ON CONFLICT (person_age, person_gender, person_education, person_income, 
                                           person_emp_exp, person_home_ownership, loan_amnt, loan_intent, 
                                           loan_int_rate, loan_percent_income, cb_person_cred_hist_length, 
                                           credit_score, previous_loan_defaults_on_file, loan_status, prediction, actual)
            DO NOTHING;
            """

            execute_values(cursor, insert_query, data_to_insert)
            conn.commit()

        print("✅ Monitoring data saved successfully.")

    except Exception as e:
        print(f"❌ Error saving monitoring data: {e}")

    finally:
        if conn:
            conn.close()


