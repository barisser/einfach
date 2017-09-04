import keras
import numpy as np
import pandas as pd

def train(model, X, scoring_function, batches=1, epochs=10):
    """
    Takes a model NN, some features X, and some scoring function,
    and stochastically reinforces "positive" outcomes.
    Breaks into N batches.
    """
    batch_fraction = 1.0 / float(batches)
    min_score = 0 # TODO revisit
    for epoch in range(epochs):
        for batch in range(batches):
            mini_X = X.sample(frac=batch_fraction)
            model = train_batch(model, mini_X, scoring_function)
    return model

def train_batch(model, mini_X, scoring_function, min_score=0):
    Y = model.predict(mini_X)
    scores = scoring_function(Y)
    total_score = scores.sum() # TODO optional exponential weighting on scores here
    print "Saw Score: {0}".format(total_score)
    if total_score > min_score:
        model.train_on_batch(X, Y)
    return model
