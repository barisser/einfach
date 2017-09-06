import math

import numpy as np

import einfach


def test_simple():
    assert True

def test_train():
    X = np.arange(100, step=0.1)
    Y = np.sin(X)
    score = lambda x: sum([r**2 for r in x])
    model = einfach.train(X, score)
    import pdb;pdb.set_trace()
