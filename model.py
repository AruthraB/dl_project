import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, Bidirectional, LSTM,
    GlobalMaxPooling1D, Dense, Dropout
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("final_data.csv")

# safety
data['text'] = data['text'].astype(str)

# ---------------- TOKENIZATION ----------------
max_words = 8000
max_len = 120

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])

X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=max_len)

# ---------------- LABEL ENCODING ----------------
cat_encoder = LabelEncoder()
urg_encoder = LabelEncoder()

y_cat = cat_encoder.fit_transform(data['category'])
y_urg = urg_encoder.fit_transform(data['urgency'])

y_cat = to_categorical(y_cat)
y_urg = to_categorical(y_urg)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
    X, y_cat, y_urg,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ----------------
input_layer = Input(shape=(max_len,))

x = Embedding(max_words, 128)(input_layer)
x = Conv1D(128, 5, activation='relu')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# OUTPUTS
category_output = Dense(y_cat.shape[1], activation='softmax', name='category')(x)
urgency_output = Dense(y_urg.shape[1], activation='softmax', name='urgency')(x)

# MODEL
model = Model(inputs=input_layer, outputs=[category_output, urgency_output])

# ---------------- COMPILE ----------------
model.compile(
    optimizer='adam',
    loss={
        'category': 'categorical_crossentropy',
        'urgency': 'categorical_crossentropy'
    },
    metrics={
        'category': ['accuracy'],
        'urgency': ['accuracy']
    }
)

# ---------------- TRAIN ----------------
model.fit(
    X_train,
    {
        'category': y_cat_train,
        'urgency': y_urg_train
    },
    validation_data=(
        X_test,
        {
            'category': y_cat_test,
            'urgency': y_urg_test
        }
    ),
    epochs=8,
    batch_size=32
)

# ---------------- SAVE ----------------
model.save("model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("cat_encoder.pkl", "wb") as f:
    pickle.dump(cat_encoder, f)

with open("urg_encoder.pkl", "wb") as f:
    pickle.dump(urg_encoder, f)

print("\n✅ MODEL SAVED (CATEGORY + URGENCY)")