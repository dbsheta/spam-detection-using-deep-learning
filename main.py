from keras.utils import to_categorical
from keras.preprocessing import sequence
from mxnet import gluon
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle

MAX_WORDS_IN_SEQ = 1000
EMBED_DIM = 100
MODEL_PATH = "models/spam_detect"

# Load Data
with open("data/dataset.pkl", 'rb') as f:
    sequences, labels, word2index = pickle.load(f)

num_words = len(word2index)
print(f"Found {num_words} unique tokens")

data = sequence.pad_sequences(sequences, maxlen=MAX_WORDS_IN_SEQ, padding='post', truncating='post')
print(labels[:10])
labels = to_categorical(labels)
print(labels[:10])

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Building the model
input_seq = Input(shape=[MAX_WORDS_IN_SEQ, ], dtype='int32')
embed_seq = Embedding(num_words + 1, EMBED_DIM, embeddings_initializer='glorot_normal', input_length=MAX_WORDS_IN_SEQ)(
    input_seq)
conv_1 = Conv1D(128, 5, activation='relu')(embed_seq)
conv_1 = MaxPooling1D(pool_size=5)(conv_1)
conv_2 = Conv1D(128, 5, activation='relu')(conv_1)
conv_2 = MaxPooling1D(pool_size=5)(conv_2)
conv_3 = Conv1D(128, 5, activation='relu')(conv_2)
conv_3 = MaxPooling1D(pool_size=35)(conv_3)
flat = Flatten()(conv_3)
flat = Dropout(0.25)(flat)
fc1 = Dense(128, activation='relu')(flat)
dense_1 = Dropout(0.25)(flat)
fc2 = Dense(2, activation='softmax')(fc1)

model = Model(input_seq, fc2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)],
    validation_data=[x_test, y_test]
)

model.save(MODEL_PATH)


class CnnClassifierModel(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(CnnClassifierModel, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv1D()

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass
