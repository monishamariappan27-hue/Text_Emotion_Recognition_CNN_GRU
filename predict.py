# ==========================================================
# Prediction Script - Text Emotion Recognition
# ==========================================================

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# Load Model + Tokenizer + Label Encoder
# ==========================================================

model = load_model("models/emotion_model.keras")
tokenizer = joblib.load("models/tokenizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

MAX_LENGTH = 100

# ==========================================================
# Prediction Function
# ==========================================================

def predict_emotion(text):

    text = text.lower()

    # stronger negation handling
    if "not happy" in text or "not good" in text or "not fine" in text:
        return "sadness", 0.95

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post")

    pred = model.predict(padded, verbose=0)

    idx = np.argmax(pred)
    emotion = label_encoder.inverse_transform([idx])[0]
    confidence = float(np.max(pred))

    return emotion, confidence


# ==========================================================
# Test Example (optional)
# ==========================================================

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or type 'exit'): ")

        if text.lower() == "exit":
            break

        emotion, confidence = predict_emotion(text)

        print("\nPredicted Emotion:", emotion)
        print("Confidence:", round(confidence * 100, 2), "%")