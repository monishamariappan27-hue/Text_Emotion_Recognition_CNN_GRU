# 🧠 Text Emotion Recognition using CNN + Bidirectional GRU

A Deep Learning-based Text Emotion Recognition system that classifies user-entered text into one of six emotions using a hybrid **Convolutional Neural Network (CNN)** and **Bidirectional Gated Recurrent Unit (BiGRU)** model. The project also includes an interactive **Streamlit web application** for real-time emotion prediction.

---

## 📌 Project Overview

Understanding emotions expressed in text is an important task in Natural Language Processing (NLP). This project uses Deep Learning techniques to classify text into different emotional categories.

The application accepts user input, preprocesses the text, predicts the emotion, displays the confidence score, visualizes prediction probabilities, and stores recent prediction history.

---

## 🎯 Objectives

- Develop a Deep Learning model for text emotion classification.
- Combine CNN and Bidirectional GRU to improve feature extraction and contextual understanding.
- Build an interactive Streamlit application for real-time prediction.
- Visualize prediction confidence and emotion probabilities.

---

## 😊 Supported Emotions

- 😊 Joy
- 😢 Sadness
- 😠 Anger
- 😨 Fear
- ❤️ Love
- 😲 Surprise

---

# 🏗 Project Architecture

```
User Text
     │
     ▼
Text Preprocessing
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
CNN Layer
     │
     ▼
Bidirectional GRU
     │
     ▼
Dense Layers
     │
     ▼
Softmax Output
     │
     ▼
Predicted Emotion
```

---

# 🧠 Deep Learning Model

The model consists of:

- Embedding Layer
- Convolutional Neural Network (CNN)
- Max Pooling Layer
- Bidirectional GRU
- Dense Layer
- Softmax Output Layer

---

# 📂 Project Structure

```
Text_Emotion_Recognition_CNN_GRU/
│
├── dataset/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
│
├── models/
│   ├── emotion_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── accuracy.png
│   └── loss.png
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The dataset contains text samples labeled with one of six emotions.

### Emotion Classes

| Emotion | Description |
|----------|-------------|
| Joy | Happiness and excitement |
| Sadness | Feeling unhappy or disappointed |
| Anger | Feeling frustrated or irritated |
| Fear | Feeling afraid or anxious |
| Love | Affection and care |
| Surprise | Unexpected reaction |

---

# ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/monishamariappan27-hue/Text_Emotion_Recognition_CNN_GRU.git
```

Move into the project folder

```bash
cd Text_Emotion_Recognition_CNN_GRU
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python train.py
```

---

# ▶️ Predict from Terminal

```bash
python predict.py
```

---

# ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 💻 Application Features

- Real-time emotion prediction
- Confidence score
- Top emotion probabilities
- Emotion probability visualization
- Prediction history
- Interactive Streamlit dashboard
- User-friendly interface

---

# 📈 Training Graphs

The project automatically generates:

- Accuracy Curve
- Loss Curve

These graphs are stored inside the **models** folder.

---

# 📷 Project Screenshots

### Home Page

(Add Screenshot)

### Prediction Page

(Add Screenshot)

### Probability Chart

(Add Screenshot)

---

# 📊 Future Improvements

- Improve prediction accuracy
- Add Attention Mechanism
- Integrate BERT model
- Speech-to-Text Emotion Recognition
- Multilingual Emotion Detection
- Cloud Deployment

---

# 🎓 Learning Outcomes

Through this project, I gained hands-on experience in:

- Natural Language Processing
- Text Preprocessing
- Tokenization
- Deep Learning
- CNN
- Bidirectional GRU
- Multi-class Classification
- Model Deployment
- Streamlit Development

---


## 📄 License

This project is developed for educational and learning purposes.
