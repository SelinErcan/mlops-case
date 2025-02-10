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

def save_to_table(df, table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cursor:
            data_to_insert = df.values.tolist()
            column_list_str = ', '.join(df.columns)

            insert_query = f"""
            INSERT INTO {table_name} ({column_list_str})
            VALUES %s
            ON CONFLICT ({column_list_str})
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


