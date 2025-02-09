import pickle
import tensorflow as tf

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