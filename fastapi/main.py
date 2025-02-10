from fastapi import FastAPI, UploadFile, File, Response
from prometheus_client import Counter, Gauge, Summary, generate_latest
from io import StringIO
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil
import asyncio
import logging
import data_drift
import model_run
import check_database as db
import uvicorn

app = FastAPI()

API_LATENCY = Summary('loan_prediction_api_latency_seconds', 'Latency of loan prediction API requests')
REQUEST_ERRORS = Counter('loan_prediction_errors_total', 'Total number of errors')
CPU_USAGE = Gauge('cpu_usage', 'CPU usage in percentage')
MEMORY_USAGE = Gauge('memory_usage', 'Memory usage in percentage')
DATA_DRIFT_DETECTED = Counter('data_drift_detected', 'Count of data drift events detected')

async def collect_system_metrics():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        await asyncio.sleep(60)  

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(collect_system_metrics())

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with API_LATENCY.time():

            model, scaler, encoder = model_run.load_model()
            train_df = db.fetch_data() 
            file_content = await file.read()
            test_df = pd.read_csv(StringIO(file_content.decode('utf-8')))

            if 'loan_status' in test_df.columns:
                y_test = test_df['loan_status']
                test_df = test_df.drop(columns=['loan_status'])
            else:
                y_test = None  

            numeric_columns = test_df.select_dtypes(include=["number"]).columns.tolist()
            categorical_columns = test_df.select_dtypes(exclude=["number"]).columns.tolist()

            drift_detected = False
            drift_results = []
            for column in numeric_columns:
                drift_message = data_drift.check_data_drift(train_df[column], test_df[column], column)
                drift_results.append({"feature": column, "drift_detected": bool(drift_message)})
                
                if drift_message:
                    DATA_DRIFT_DETECTED.inc()
                    logging.info(f"Data drift detected in column '{column}': {drift_message}")
                    drift_detected = True
                else:
                    logging.info(f"No drift detected in column '{column}'.")

            if drift_detected:
                logging.info("Triggering model retraining...")
                model_run.retrain_model(train_df)


            test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])  
            encoded_test = encoder.transform(test_df[categorical_columns])  
            
            encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns))
            X_test_encoded = pd.concat([test_df[numeric_columns].reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)
            X_test_encoded = X_test_encoded.values

            predictions_prob = model.predict(X_test_encoded)
            predictions = (predictions_prob > 0.5).astype(int).flatten().tolist()

            monitoring_data = test_df.copy()
            monitoring_data["prediction"] = predictions
            if y_test is not None:
                monitoring_data["actual"] = y_test.tolist()
            else:
                monitoring_data["actual"] = None  

            db.save_to_table(monitoring_data, table_name="monitoring") 

            result = {"drift_results": drift_results}

            if y_test is not None:
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)

                result.update({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

            return result

    except Exception as e:
        REQUEST_ERRORS.inc()
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}


@app.get("/metrics")
def metrics():
    content = generate_latest()
    return Response(content, media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)