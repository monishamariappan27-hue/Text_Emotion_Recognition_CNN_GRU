
# ==========================================================
# Emotion Recognition AI Dashboard
# CNN + Bidirectional GRU
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Emotion Recognition AI", page_icon="🧠", layout="wide")

@st.cache_resource
def load_assets():
    model = load_model("models/emotion_model.keras")
    tokenizer = joblib.load("models/tokenizer.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, tokenizer, encoder

model, tokenizer, label_encoder = load_assets()
MAX_LENGTH = 100

if "history" not in st.session_state:
    st.session_state.history = []

emoji = {
    "joy":"😊",
    "sadness":"😢",
    "anger":"😠",
    "fear":"😨",
    "love":"❤️",
    "surprise":"😲"
}

desc = {
    "joy":"Positive emotion expressing happiness.",
    "sadness":"Feeling of disappointment or grief.",
    "anger":"Strong feeling of annoyance or frustration.",
    "fear":"Feeling of worry or danger.",
    "love":"Affection and care.",
    "surprise":"Unexpected reaction."
}

def clean(txt):
    txt = txt.lower()
    return re.sub(r"[^a-z\s]","",txt)

def predict(text):
    seq = tokenizer.texts_to_sequences([clean(text)])
    pad = pad_sequences(seq,maxlen=MAX_LENGTH,padding="post")
    probs = model.predict(pad,verbose=0)[0]
    idx = np.argmax(probs)
    emo = label_encoder.inverse_transform([idx])[0]
    return emo,float(probs[idx]),probs

st.sidebar.title("🧠 Emotion AI")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home","😊 Predict","📊 Analytics","📜 History","ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** CNN + Bidirectional GRU")
st.sidebar.write("**Classes:**")
for e in label_encoder.classes_:
    st.sidebar.write(f"- {e}")

if page=="🏠 Home":
    st.title("🧠 Emotion Recognition AI")
    st.markdown("### Deep Learning based Text Emotion Detection")
    c1,c2,c3=st.columns(3)
    c1.metric("Supported Emotions",len(label_encoder.classes_))
    c2.metric("Model","CNN + BiGRU")
    c3.metric("Input","Text")
    st.info("Use the sidebar to navigate through the application.")

elif page=="😊 Predict":
    st.title("😊 Predict Emotion")
    text = st.text_area("Enter text",height=150)
    if st.button("Predict"):
        if text.strip():
            emo,conf,probs=predict(text)
            st.session_state.history.append([text,emo,conf])
            st.success(f"Prediction: {emo} {emoji.get(emo,'')}")
            st.progress(conf)
            st.write(f"**Confidence:** {conf*100:.2f}%")
            st.write(desc.get(emo,""))
            st.subheader("Top 3 Predictions")
            top=np.argsort(probs)[-3:][::-1]
            for i in top:
                name=label_encoder.inverse_transform([i])[0]
                st.write(f"**{name} {emoji.get(name,'')}**")
                st.progress(float(probs[i]))
                st.caption(f"{probs[i]*100:.2f}%")
            st.subheader("Probability Distribution")
            chart={label_encoder.classes_[i]:float(probs[i]) for i in range(len(probs))}
            st.bar_chart(chart)
        else:
            st.warning("Please enter text.")

elif page=="📊 Analytics":
    st.title("📊 Analytics")
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Accuracy")
        try:
            st.image("models/accuracy.png",use_container_width=True)
        except:
            st.warning("accuracy.png not found.")
    with col2:
        st.subheader("Loss")
        try:
            st.image("models/loss.png",use_container_width=True)
        except:
            st.warning("loss.png not found.")

elif page=="📜 History":
    st.title("📜 Prediction History")
    if st.session_state.history:
        df=pd.DataFrame(st.session_state.history,columns=["Text","Emotion","Confidence"])
        st.dataframe(df,use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv"
        )
        if st.button("🗑 Clear History"):
            st.session_state.history=[]
            st.rerun()
    else:
        st.info("No predictions yet.")

else:
    st.title("ℹ️ About")
    st.markdown("""
### Emotion Recognition AI

**Deep Learning Model**
- CNN
- Bidirectional GRU

**Supported Emotions**
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

This application predicts emotions from user-entered text.
""")

st.markdown("---")
st.caption("Built with Streamlit • TensorFlow • CNN + Bidirectional GRU")
