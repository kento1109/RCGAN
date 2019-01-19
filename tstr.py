"""
Train on synthetic, test on real
evaluate generated samples
"""

import numpy as np
import os
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

####### edit this variable depending on the situation ##########
REAL_FILE_NAME = 'inputs/sin_wave.npz'  # real data for mmd
GENERATED_FILE_NAME = 'outputs/samples.npz'  # generated samples
num_classes = 3
num_epochs = 10
batch_size = 32
hidden_dim = 128
################################################################


def build_model(seq_length, input_dim):
    model = Sequential()
    model.add(LSTM(units=hidden_dim,
                   batch_input_shape=(None, seq_length, input_dim),
                   return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    
    # choose GPU devise
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # load samples
    npz = np.load(GENERATED_FILE_NAME)
    X_train, y_train = npz['x'], npz['y']
    npz = np.load(REAL_FILE_NAME)
    X_test, y_test = npz['x'], npz['y']

    # to one-hot vector
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = build_model(seq_length=X_train.shape[1], input_dim=X_train.shape[2])
    early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1)

    fit = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_data=[X_test, y_test],
                    callbacks=[early_stopping])


