import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.optimizers import RMSprop

def build_nn():
    model = Sequential()
    #model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(16, input_dim=1))
    #model.add(Dense(128))
    #model.add(Dense(16))
    model.add(Dense(1))
    #opt = Adam()  # ???
    model.compile(loss='mse', optimizer='adam')
    return model

def train(model, X, scoring_function, batches=1, epochs=10):
    """
    Takes a model NN, some features X, and some scoring function,
    and stochastically reinforces "positive" outcomes.
    Breaks into N batches.

    Scoring Functions must always return AVERAGE scores that do not differentiate between batch lengths.
    """
    batch_fraction = 1.0 / float(batches)
    for epoch in range(epochs):
        avg_score = scoring_function(model.predict(X))
        for batch in range(batches):
            mini_X = np.random.choice(X, size=int(batch_fraction * len(X)))
            model = train_batch(model, mini_X, scoring_function, avg_score)
    return model

def train_batch(model, mini_X, scoring_function, avg_score):
    Y = model.predict(mini_X)
    score = scoring_function(Y)
    print "Saw Score: {0}".format(score)
    weights = (np.ones((1, len(mini_X))) * (score - avg_score))[0]
    model.train_on_batch(mini_X, Y, sample_weight=weights)
    return model
