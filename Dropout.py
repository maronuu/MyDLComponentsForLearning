import numpy as np


class Dropout:

    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.drop_mask = None

    def forward(self, x, is_train=True):
        if is_train:
            self.drop_mask = np.random.rand(*x.shape) > self.ratio
            return x * self.drop_mask
        else:
            return x * (1. - self.ratio)
    
    def backward(self, dout):
        return dout * self.drop_mask
