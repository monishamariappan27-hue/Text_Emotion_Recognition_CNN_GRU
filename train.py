# ==========================================================
# Text Emotion Recognition using DistilBERT (fine-tuned)
# train.py
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

# ==========================================================
# Config
# ==========================================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
OUTPUT_DIR = "models/distilbert_emotion"
EPOCHS = 3
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load Dataset (same files as before)
# ==========================================================

train_df = pd.read_csv("dataset/train.txt", sep=";", names=["text", "emotion"])
test_df = pd.read_csv("dataset/test.txt", sep=";", names=["text", "emotion"])
val_df = pd.read_csv("dataset/val.txt", sep=";", names=["text", "emotion"])

print("\nDataset Loaded Successfully")
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("Val Shape:", val_df.shape)

# ==========================================================
# Encode Labels
# ==========================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_df["emotion"])
y_val = label_encoder.transform(val_df["emotion"])
y_test = label_encoder.transform(test_df["emotion"])

print("\nEmotion Classes:")
print(label_encoder.classes_)

joblib.dump(label_encoder, f"{OUTPUT_DIR}/label_encoder.pkl")

NUM_LABELS = len(label_encoder.classes_)

# ==========================================================
# Tokenizer
# ==========================================================

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

train_encodings = tokenize(train_df["text"])
val_encodings = tokenize(val_df["text"])
test_encodings = tokenize(test_df["text"])

# ==========================================================
# Dataset class
# ==========================================================

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

train_dataset = EmotionDataset(train_encodings, y_train)
val_dataset = EmotionDataset(val_encodings, y_val)
test_dataset = EmotionDataset(test_encodings, y_test)

# ==========================================================
# Model
# ==========================================================

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

# ==========================================================
# Metrics
# ==========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ==========================================================
# Training arguments (CPU-friendly)
# ==========================================================

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    no_cuda=True,  # force CPU; this is expected to be slower than GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ==========================================================
# Train
# ==========================================================

print("\nStarting training (CPU) — this will take a while...\n")
trainer.train()

# ==========================================================
# Evaluate on test set
# ==========================================================

test_results = trainer.evaluate(test_dataset)
print("\nTest Results:", test_results)

# ==========================================================
# Save final model + tokenizer
# ==========================================================

final_path = f"{OUTPUT_DIR}/saved"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"\nModel and tokenizer saved to {final_path}")
print("\nTraining Completed Successfully 🚀")