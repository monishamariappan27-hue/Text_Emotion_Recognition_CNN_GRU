# ==========================================================
# Text Emotion Recognition AI Dashboard
# CNN + Bidirectional GRU
# Developed by Monisha Mariappan
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Emotion Recognition AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

.block-container{
    padding-top:2rem;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#2563eb;
    text-align:center;
}

.subtitle{
    font-size:18px;
    color:#475569;
    text-align:center;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,.08);
    border-left:6px solid #2563eb;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

div.stButton > button{
    width:100%;
    height:50px;
    font-size:18px;
    border-radius:12px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL (this section was missing from the broken file)
# ==========================================================

@st.cache_resource
def load_assets():
    model = load_model("models/emotion_model.keras")
    tokenizer = joblib.load("models/tokenizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_assets()

MAX_LENGTH = 100

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# EMOJIS
# ==========================================================

emoji_dict = {
    "joy":"😊",
    "sadness":"😢",
    "anger":"😠",
    "fear":"😨",
    "love":"❤️",
    "surprise":"😲"
}

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_emotion(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post")
    probs = model.predict(padded, verbose=0)[0]
    index = np.argmax(probs)
    emotion = label_encoder.inverse_transform([index])[0]
    confidence = float(probs[index])
    return emotion, confidence, probs

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🧠 Emotion AI")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "😊 Predict Emotion",
        "📊 Analytics",
        "📜 Prediction History",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Model Loaded")

st.sidebar.info("CNN + Bidirectional GRU")

st.sidebar.markdown("### 😊 Supported Emotions")

for emotion in label_encoder.classes_:
    st.sidebar.write(f"{emoji_dict.get(emotion,'🧠')} {emotion.title()}")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by")
st.sidebar.write("**Monisha Mariappan**")

# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if page == "🏠 Dashboard":

    st.markdown("<h1 class='title'>🧠 Text Emotion Recognition AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Deep Learning using CNN + Bidirectional GRU</p>", unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🧠 Model", "CNN + BiGRU")

    with c2:
        st.metric("😊 Emotions", len(label_encoder.classes_))

    with c3:
        st.metric("📄 Input", "Text")

    with c4:
        st.metric("⚡ Framework", "TensorFlow")

    st.write("")

    left, right = st.columns([2,1])

    with left:
        st.markdown("""
<div class="card">
<h2>📌 About Project</h2>
This project predicts the emotion present in a text sentence using a Deep Learning model.
The model combines:
✅ Convolutional Neural Network (CNN)
✅ Bidirectional GRU
to understand both local word patterns and contextual information.
The application is built using Streamlit and provides real-time emotion prediction.
</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown("""
<div class="card">
<h3>🎯 Supported Emotions</h3>
😊 Joy
😢 Sadness
😠 Anger
😨 Fear
❤️ Love
😲 Surprise
</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("⚙️ How It Works")

    st.code("""
User Text
     │
     ▼
Text Cleaning
     │
     ▼
Tokenizer
     │
     ▼
Padding
     │
     ▼
Embedding Layer
     │
     ▼
CNN
     │
     ▼
Bidirectional GRU
     │
     ▼
Dense Layer
     │
     ▼
Softmax
     │
     ▼
Emotion Prediction
""")

    st.write("")

    st.subheader("✨ Features")

    f1, f2 = st.columns(2)

    with f1:
        st.success("✅ Real-Time Emotion Prediction")
        st.success("✅ Confidence Score")
        st.success("✅ Top 3 Predictions")
        st.success("✅ Emotion Probability Chart")

    with f2:
        st.success("✅ Analytics Dashboard")
        st.success("✅ Prediction History")
        st.success("✅ Download Prediction History")
        st.success("✅ CNN + BiGRU Model")

    st.write("")

    st.info("👉 Use the sidebar and open **😊 Predict Emotion** to classify your text.")

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "😊 Predict Emotion":

    st.markdown("<h1 class='title'>😊 Emotion Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter any sentence and let the AI detect the emotion.</p>", unsafe_allow_html=True)

    st.write("")

    user_input = st.text_area(
        "✍ Enter Text",
        height=180,
        placeholder="Example: I am feeling very happy today!"
    )

    col1, col2 = st.columns([3,1])

    with col1:
        predict_btn = st.button("🚀 Predict Emotion", use_container_width=True)

    with col2:
        clear_btn = st.button("🗑 Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if predict_btn:

        if user_input.strip() == "":
            st.warning("⚠ Please enter some text.")
        else:
            emotion, confidence, probs = predict_emotion(user_input)

            st.session_state.history.append((user_input, emotion, confidence))

            emoji = emoji_dict.get(emotion, "🧠")

            descriptions = {
                "joy":"Feeling happy, cheerful and excited.",
                "sadness":"Feeling unhappy, disappointed or emotional.",
                "anger":"Feeling frustrated, irritated or annoyed.",
                "fear":"Feeling anxious, scared or nervous.",
                "love":"Feeling affection, kindness and care.",
                "surprise":"Unexpected emotion or astonishment."
            }

            st.write("")

            left, right = st.columns([2,1])

            with left:
                st.markdown(f"""
                <div class="card">
                <h2>🎯 Prediction Result</h2>
                <h1 style="color:#2563eb;">
                {emoji} {emotion.title()}
                </h1>
                <h3>
                Confidence :
                {confidence*100:.2f}%
                </h3>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(confidence))

                if confidence >= 0.90:
                    st.success("Excellent Confidence")
                elif confidence >= 0.75:
                    st.info("Good Confidence")
                else:
                    st.warning("Low Confidence")

            with right:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📖 Emotion")
                st.write(descriptions[emotion])
                st.markdown("</div>", unsafe_allow_html=True)

            st.write("")

            st.subheader("🏆 Top 3 Predictions")

            top3 = probs.argsort()[-3:][::-1]

            for i in top3:
                e = label_encoder.inverse_transform([i])[0]
                st.write(f"### {emoji_dict.get(e,'')} {e.title()}")
                st.progress(float(probs[i]))
                st.caption(f"{probs[i]*100:.2f}%")

            st.write("")

            st.subheader("📊 Emotion Probability")

            chart = pd.DataFrame({
                "Emotion":label_encoder.classes_,
                "Probability":probs
            })

            st.bar_chart(chart.set_index("Emotion"))

            st.write("")

            st.subheader("📋 Probability Table")

            table = pd.DataFrame({
                "Emotion":label_encoder.classes_,
                "Probability (%)":[round(i*100,2) for i in probs]
            })

            st.dataframe(table, use_container_width=True)

            st.write("")

            with st.expander("🧠 Model Information"):
                st.write("""
**Model Used**

- CNN
- Bidirectional GRU

**Framework**

- TensorFlow
- Keras

**Input**

- Text Sentence

**Output**

- Emotion Class
                """)

# ==========================================================
# ANALYTICS PAGE
# ==========================================================

elif page == "📊 Analytics":

    st.markdown("<h1 class='title'>📊 Model Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Performance Analysis of CNN + Bidirectional GRU Model</p>", unsafe_allow_html=True)

    st.write("")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("🧠 Model", "CNN + BiGRU")

    with m2:
        st.metric("😊 Classes", "6")

    with m3:
        st.metric("📄 Max Length", "100")

    with m4:
        st.metric("⚡ Framework", "TensorFlow")

    st.write("")

    st.subheader("📈 Training Performance")

    left, right = st.columns(2)

    with left:
        st.markdown("### Accuracy")
        try:
            st.image("models/accuracy.png", use_container_width=True)
        except Exception:
            st.warning("accuracy.png not found.")

    with right:
        st.markdown("### Loss")
        try:
            st.image("models/loss.png", use_container_width=True)
        except Exception:
            st.warning("loss.png not found.")

    st.write("")

    st.subheader("📂 Dataset Statistics")

    dataset = pd.DataFrame({
        "Emotion":["Joy","Sadness","Anger","Fear","Love","Surprise"],
        "Training Samples":[5362,4666,2159,1937,1304,572]
    })

    st.dataframe(dataset, use_container_width=True)

    st.bar_chart(dataset.set_index("Emotion"))

    st.write("")

    st.subheader("🥧 Emotion Distribution")

    pie = dataset.set_index("Emotion")

    st.pyplot(
        pie.plot.pie(
            y="Training Samples",
            figsize=(6,6),
            autopct="%1.1f%%",
            legend=False
        ).figure
    )

    st.write("")

    st.subheader("🧠 Deep Learning Architecture")

    st.code("""

                 Input Text
                      │
                      ▼
             Text Preprocessing
                      │
                      ▼
                 Tokenization
                      │
                      ▼
                   Padding
                      │
                      ▼
               Embedding Layer
                      │
                      ▼
              Convolutional Layer
                      │
                      ▼
                 MaxPooling
                      │
                      ▼
             Bidirectional GRU
                      │
                      ▼
                 Dense Layer
                      │
                      ▼
               Softmax Output
                      │
                      ▼
             Predicted Emotion

""")

    st.info("CNN extracts local textual features while Bidirectional GRU captures long-term contextual dependencies from both forward and backward directions.")

    st.write("")

    st.subheader("✨ Model Highlights")

    c1, c2 = st.columns(2)

    with c1:
        st.success("✅ Deep Learning")
        st.success("✅ CNN Feature Extraction")
        st.success("✅ Bidirectional GRU")
        st.success("✅ Multi-Class Classification")

    with c2:
        st.success("✅ TensorFlow")
        st.success("✅ Streamlit Deployment")
        st.success("✅ Probability Prediction")
        st.success("✅ Interactive Dashboard")

    st.write("")

    st.subheader("📋 Performance Summary")

    summary = pd.DataFrame({
        "Property":["Model","Input","Output","Framework","Deep Learning Layers","Deployment"],
        "Value":["CNN + Bidirectional GRU","Text","Emotion","TensorFlow","CNN + BiGRU + Dense","Streamlit"]
    })

    st.table(summary)

    st.success("✔ Model loaded successfully and ready for prediction.")

# ==========================================================
# PREDICTION HISTORY
# ==========================================================

elif page == "📜 Prediction History":

    st.markdown("<h1 class='title'>📜 Prediction History</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>View, search, analyze and download previous predictions.</p>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No prediction history available.")
    else:
        history_df = pd.DataFrame(
            st.session_state.history,
            columns=["Input Text","Emotion","Confidence"]
        )

        history_df["Confidence"] = (history_df["Confidence"] * 100).round(2)

        search = st.text_input("🔍 Search Text")

        if search:
            history_df = history_df[
                history_df["Input Text"].str.contains(search, case=False, na=False)
            ]

        total_predictions = len(history_df)
        average_confidence = history_df["Confidence"].mean()
        top_emotion = history_df["Emotion"].mode()[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("📄 Total Predictions", total_predictions)

        with c2:
            st.metric("🎯 Avg Confidence", f"{average_confidence:.2f}%")

        with c3:
            st.metric("😊 Most Predicted", top_emotion.title())

        st.write("")

        st.subheader("📋 Prediction Records")

        st.dataframe(history_df, use_container_width=True, height=350)

        st.write("")

        st.subheader("📊 Emotion Frequency")

        emotion_count = history_df["Emotion"].value_counts()

        st.bar_chart(emotion_count)

        st.write("")

        st.subheader("🥧 Emotion Distribution")

        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(emotion_count, labels=emotion_count.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        st.write("")

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.write("")

        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("Prediction history cleared successfully.")
            st.rerun()

        st.write("")

        st.subheader("🕒 Recent Predictions")

        recent = history_df.tail(5)

        st.table(recent)

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    st.markdown("<h1 class='title'>ℹ About This Project</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Text Emotion Recognition using CNN + Bidirectional GRU</p>", unsafe_allow_html=True)

    st.write("")

    st.subheader("👩‍💻 Developer")

    st.success("""
### Monisha Mariappan

🎓 B.Tech - Artificial Intelligence and Data Science

🏫 Velalar College of Engineering and Technology

💻 Full Stack Web Developer | Machine Learning | Deep Learning Enthusiast
""")

    st.write("")

    st.subheader("📌 Project Description")

    st.write("""
This project is a **Deep Learning based Text Emotion Recognition System**
that predicts the emotional state of a sentence.

The model combines the strengths of:

- 🧠 Convolutional Neural Network (CNN)
- 🔄 Bidirectional Gated Recurrent Unit (BiGRU)

to capture both local textual features and contextual relationships.

The application is deployed using **Streamlit** for an interactive user experience.
""")

    st.write("")

    st.subheader("✨ Features")

    c1, c2 = st.columns(2)

    with c1:
        st.success("✅ Real-Time Emotion Prediction")
        st.success("✅ Top-3 Emotion Prediction")
        st.success("✅ Prediction Confidence")
        st.success("✅ Interactive Dashboard")
        st.success("✅ Analytics Dashboard")

    with c2:
        st.success("✅ Prediction History")
        st.success("✅ Download CSV")
        st.success("✅ Dataset Visualization")
        st.success("✅ Deep Learning Model")
        st.success("✅ Responsive UI")

    st.write("")

    st.subheader("🧠 Deep Learning Architecture")

    st.code("""

                 User Input
                      │
                      ▼
              Text Cleaning
                      │
                      ▼
                Tokenization
                      │
                      ▼
                   Padding
                      │
                      ▼
               Embedding Layer
                      │
                      ▼
                      CNN
                      │
                      ▼
                 Max Pooling
                      │
                      ▼
             Bidirectional GRU
                      │
                      ▼
                 Dense Layer
                      │
                      ▼
                 Softmax Layer
                      │
                      ▼
            Emotion Classification

""")

    st.write("")

    st.subheader("🛠 Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.info("🐍 Python")
        st.info("🧠 TensorFlow")
        st.info("⚡ Keras")

    with tech2:
        st.info("📊 Pandas")
        st.info("🔢 NumPy")
        st.info("📈 Matplotlib")

    with tech3:
        st.info("🤖 Scikit-Learn")
        st.info("🌐 Streamlit")
        st.info("💾 Joblib")

    st.write("")

    st.subheader("😊 Supported Emotions")

    emotions = pd.DataFrame({
        "Emotion": ["😊 Joy","😢 Sadness","😠 Anger","😨 Fear","❤️ Love","😲 Surprise"]
    })

    st.table(emotions)

    st.write("")

    st.subheader("🚀 Future Enhancements")

    st.write("""
- 🔹 Improve prediction accuracy using Attention Mechanism.
- 🔹 Integrate BERT or RoBERTa models.
- 🔹 Add multilingual emotion recognition.
- 🔹 Build a Flask REST API.
- 🔹 Deploy on Streamlit Cloud or Hugging Face Spaces.
- 🔹 Add Speech-to-Text emotion recognition.
- 🔹 Mobile application support.
""")

    st.write("")

    st.subheader("📊 Project Information")

    info = pd.DataFrame({
        "Property":["Deep Learning Model","Framework","Input","Output","Emotion Classes","Deployment"],
        "Value":["CNN + Bidirectional GRU","TensorFlow","Text","Emotion","6","Streamlit"]
    })

    st.table(info)

    st.write("")

    st.subheader("🔗 GitHub Repository")

    st.code("https://github.com/monishamariappan27-hue/Text_Emotion_Recognition_CNN_GRU")

    st.write("")

    st.markdown("---")

    st.markdown("""
<div style='text-align:center;padding:20px;background:#f8fafc;border-radius:12px;'>
<h2>🧠 Text Emotion Recognition AI</h2>
<h4>CNN + Bidirectional GRU</h4>
<p>Developed by <b>Monisha Mariappan</b></p>
<p>Made with ❤️ using Python, TensorFlow & Streamlit</p>
⭐ Thank you for visiting this project.
</div>
""", unsafe_allow_html=True)
