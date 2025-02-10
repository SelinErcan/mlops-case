import pickle
import tensorflow as tf
import check_database as db
from sklearn.preprocessing import  OneHotEncoder, RobustScaler
import pandas as pd
import numpy as np
from scipy.stats import zscore
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder, RobustScaler
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras

def load_model():
    try:
        with open("model/scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        with open("model/encoder.pkl", "rb") as encoder_file:
            encoder = pickle.load(encoder_file)

        model = tf.keras.models.load_model("model/model.h5")

        print("✅ Model, scaler, and encoder loaded successfully.")

        return model, scaler, encoder

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def build_model(input_shape):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,)),  
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")  
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def retrain_model(train_df):
    print("Model is retraining...")

    print("\tProcessing data...")

    monitoring_data = db.fetch_data(query="SELECT * FROM monitoring")
    monitoring_data["loan_status"] = monitoring_data["actual"].where(monitoring_data["actual"].notna(), monitoring_data["prediction"])

    selected_columns = [
        "person_gender", "person_education", "person_income", "person_home_ownership",
        "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income", 
        "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file", "loan_status"
    ]

    train_df = train_df[selected_columns]
    monitoring_data = monitoring_data[selected_columns]

    train_df = pd.concat([train_df, monitoring_data]).drop_duplicates().reset_index(drop=True)

    X_train = train_df.drop('loan_status', axis=1)
    y_train = train_df['loan_status']  

    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    print("numeric_columns",numeric_columns)
    print("categorical_columns",categorical_columns)
    print(X_train.dtypes)

    scaler = RobustScaler(quantile_range=(20, 80))

    encoder = OneHotEncoder(sparse_output=False, drop='first')

    encoded_train_categories = encoder.fit_transform(X_train[categorical_columns])

    encoded_train_df = pd.DataFrame(encoded_train_categories, columns=encoder.get_feature_names_out(categorical_columns))

    X_train_encoded = pd.concat([X_train[numeric_columns].reset_index(drop=True), encoded_train_df.reset_index(drop=True)], axis=1)
    

    X_train_encoded = X_train_encoded.values

    y_train = np.array(y_train)

    print(f"X_train_encoded shape: {X_train_encoded.shape}")
    print(f"y_train shape: {y_train.shape}")

    z_scores = np.abs(zscore(X_train_encoded))

    threshold = 3

    mask = (z_scores < threshold).all(axis=1)  
    X_train_encoded = X_train_encoded[mask]
    y_train = y_train[mask]

    print("\tmodel training...")

    final_model = build_model(input_shape = X_train_encoded.shape[1])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    final_model.fit(X_train_encoded, y_train, epochs=20, batch_size=132, validation_split=0.2, callbacks=[reduce_lr])

    y_train_pred_prob = final_model.predict(X_train_encoded)
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)  

    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    print("\nModel Evaluation on Training Set:")
    print(f"Accuracy:  {accuracy_train:.4f}")
    print(f"Precision: {precision_train:.4f}")
    print(f"Recall:    {recall_train:.4f}")
    print(f"F1-score:  {f1_train:.4f}")

    mlflow.set_experiment("Loan Prediction Retraining")

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.keras.log_model(final_model, "model")
        
        result = mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "LoanPredictionModel")
        
        mlflow.log_metric("accuracy", accuracy_train)
        mlflow.log_metric("precision", precision_train)
        mlflow.log_metric("recall", recall_train)
        mlflow.log_metric("f1_score", f1_train)
        mlflow.log_artifact("model/scaler.pkl")
        mlflow.log_artifact("model/encoder.pkl")


    with open('model/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open('model/encoder.pkl', 'wb') as encoder_file:
        pickle.dump(encoder, encoder_file)

    final_model.save("model/model.h5")

    print("Model retrained and deployed.")

