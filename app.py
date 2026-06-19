# ==========================================================
# Text Emotion Recognition App
# CNN + Bidirectional GRU
# ==========================================================

import streamlit as st
import numpy as np
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# Load model and assets
# ==========================================================

model = load_model("models/emotion_model.keras")
tokenizer = joblib.load("models/tokenizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

MAX_LENGTH = 100

# ==========================================================
# Page config
# ==========================================================

st.set_page_config(page_title="Emotion AI Detector", page_icon="🧠", layout="centered")

# ==========================================================
# Session state for history
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# Text cleaning
# ==========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ==========================================================
# Prediction function
# ==========================================================

def predict_emotion(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post")

    probs = model.predict(padded, verbose=0)[0]

    idx = np.argmax(probs)
    emotion = label_encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx])

    return emotion, confidence, probs

# ==========================================================
# Emoji mapping
# ==========================================================

emoji_dict = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# ==========================================================
# UI
# ==========================================================

st.title("🧠 Emotion Detection AI")
st.caption("CNN + Bidirectional GRU text emotion classifier")

with st.expander("ℹ️ Known limitation"):
    st.write(
        "This model is trained on word patterns and doesn't reliably understand "
        "negation (e.g. 'not happy' may still be read as positive due to the "
        "word 'happy'). Treat low-confidence or short sentences with caution."
    )

user_input = st.text_area("Enter your text:")

if st.button("🚀 Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        emotion, confidence, probs = predict_emotion(user_input)
        emoji = emoji_dict.get(emotion, "🧠")

        st.session_state.history.append((user_input, emotion, confidence))

        st.subheader("Result")
        st.success(f"Emotion: {emotion} {emoji}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
        st.progress(confidence)

        if confidence < 0.5:
            st.warning(
                "Low confidence — the model isn't sure. This often happens with "
                "negation, sarcasm, or mixed emotions in one sentence."
            )

        # Top 3 predictions
        st.subheader("🔥 Top 3 Predictions")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            emo = label_encoder.inverse_transform([i])[0]
            st.write(f"{emo} {emoji_dict.get(emo, '')} — {probs[i] * 100:.2f}%")

        # Full distribution chart
        st.subheader("📊 Emotion Probability Distribution")
        chart_data = {
            label_encoder.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }
        st.bar_chart(chart_data)

# ==========================================================
# History section
# ==========================================================

if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Recent Predictions")
    for t, e, c in reversed(st.session_state.history[-5:]):
        emoji = emoji_dict.get(e, "")
        st.write(f"**\"{t}\"** → {e} {emoji} ({c*100:.2f}%)")

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")
st.markdown("Built with CNN + Bidirectional GRU")