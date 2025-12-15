import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Counterfeit text descriptions
data = {
    "description": [
        "Fake branded shoes with poor stitching and misspelled logo",
        "Counterfeit handbag with wrong serial number and cheap leather",
        "Imitation perfume bottle with spelling errors on the box",
        "Duplicate sports jersey with faded colors and uneven logo",
        "Knockoff headphones that feel flimsy and lack proper branding",
        "Replica watch with incorrect date function and rough finish",
        "Counterfeit sunglasses without UV protection and crooked frame",
        "Fake mobile phone with outdated software and low-quality screen",
        "Duplicate wallet with uneven stitching and fake hologram sticker",
        "Knockoff sneakers with incorrect sole design and loose threads"
    ],
    "label": [1] * 10  # 1 = counterfeit
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
os.makedirs("data", exist_ok=True)
csv_path = "data/counterfeit_texts.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Counterfeit dataset saved at {csv_path}")

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["description"])
X = tokenizer.texts_to_sequences(df["description"])
X = pad_sequences(X, maxlen=20)

y = df["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50, input_length=20),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("\nTraining text model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

print("\nEvaluating text model...")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["counterfeit"]))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/counterfeit_text_model.h5")
print("\n✅ Counterfeit text model saved at models/counterfeit_text_model.h5")
