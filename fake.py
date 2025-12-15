import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM

# ---------------------
# CONFIG
# ---------------------
IMG_SIZE = (224, 224)
NUM_CLASSES = 2
IMG_DIR_AUTH = "data/images/authentic"
IMG_DIR_COUNT = "data/images/counterfeit"
TEXT_FILE_AUTH = "data/text/product_descriptions.csv"
TEXT_FILE_COUNT = "data/text/counterfeit_descriptions.csv"

# ---------------------
# IMAGE DATA LOADING
# ---------------------
def load_images(img_dir, label):
    images = []
    labels = []
    for file in os.listdir(img_dir):
        try:
            img_path = os.path.join(img_dir, file)
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {file}: {e}")
    return images, labels

print("Loading images...")
auth_imgs, auth_labels = load_images(IMG_DIR_AUTH, "authentic")
count_imgs, count_labels = load_images(IMG_DIR_COUNT, "counterfeit")

X_img = np.array(auth_imgs + count_imgs)
y_img = np.array(auth_labels + count_labels)

label_encoder_img = LabelEncoder()
y_img_encoded = label_encoder_img.fit_transform(y_img)

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, y_img_encoded, test_size=0.2, random_state=42
)

print(f"Loaded {len(X_img)} images.")

# ---------------------
# IMAGE MODEL
# ---------------------
print("Building image model...")
input_tensor = Input(shape=(224, 224, 3))
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model_img = Model(inputs=base_model.input, outputs=output)
model_img.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training image model...")
model_img.fit(X_train_img, y_train_img, epochs=5, validation_data=(X_test_img, y_test_img))

print("Evaluating image model...")
img_preds = np.argmax(model_img.predict(X_test_img), axis=1)
print(classification_report(y_test_img, img_preds, target_names=label_encoder_img.classes_))

# ---------------------
# TEXT DATA LOADING
# ---------------------
print("Loading text descriptions...")
auth_texts = pd.read_csv(TEXT_FILE_AUTH)["description"].tolist()
count_texts = pd.read_csv(TEXT_FILE_COUNT)["description"].tolist()

texts = auth_texts + count_texts
labels = ["authentic"] * len(auth_texts) + ["counterfeit"] * len(count_texts)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=50, padding="post", truncating="post")

label_encoder_text = LabelEncoder()
y_text_encoded = label_encoder_text.fit_transform(labels)

X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    padded_sequences, y_text_encoded, test_size=0.2, random_state=42
)

# ---------------------
# TEXT MODEL
# ---------------------
print("Building text model...")
text_input = Input(shape=(50,))
embedding = Embedding(input_dim=5000, output_dim=64)(text_input)
lstm = LSTM(64)(embedding)
dense = Dense(32, activation="relu")(lstm)
text_output = Dense(NUM_CLASSES, activation="softmax")(dense)

model_text = Model(inputs=text_input, outputs=text_output)
model_text.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training text model...")
model_text.fit(X_train_txt, y_train_txt, epochs=5, validation_data=(X_test_txt, y_test_txt))

print("Evaluating text model...")
text_preds = np.argmax(model_text.predict(X_test_txt), axis=1)
print(classification_report(y_test_txt, text_preds, target_names=label_encoder_text.classes_))

print("✅ Counterfeit detection models trained successfully!")
