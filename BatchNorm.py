import numpy as np


class BatchNorm:
    def __init__(self, gamma, beta, momentum=0.9,
                 test_mean=None, test_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.test_mean = test_mean
        self.test_var = test_var

        # forward memo
        self.batch_size = None
        self.xmu = None
        self.xhat = None
        self.ivar = None
        self.sqrtvar = None
        self.var = None
        # const
        self.eps = 1e-7

        # backprop memo
        self.dbeta = None
        self.dgamma = None

    def forward(self, x, is_train=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            # convert x into 2D
            N = x.shape[0]
            x = x.reshape(N, -1)

        if self.test_mean is None:
            _, D = x.shape
            self.test_mean = np.zeros(D)
            self.test_var = np.zeros(D)

        if is_train:
            mu = np.mean(x, axis=0)
            xmu = x - mu
            var = np.mean(xmu**2, axis=0)
            sqrtvar = np.sqrt(var + self.eps)
            ivar = 1 / sqrtvar
            xhat = xmu * ivar

            self.batch_size = x.shape[0]
            self.xmu = xmu
            self.var = var
            self.sqrtvar = sqrtvar
            self.ivar = ivar
            self.xhat = xhat
            # Momentum : (1 - Momentum) の比で更新
            self.test_mean = self.momentum * self.test_mean + \
                (1 - self.momentum) * mu
            self.test_var = self.momentum * \
                self.test_var + (1 - self.momentum) * var
        else:
            xmu = x - self.test_mean
            xhat = xmu / (np.sqrt(self.test_var + self.eps))

        out = self.gamma * xmu + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            # convert dout into 2D
            N = dout.shape[0]
            dout = dout.reshape(N, -1)

        N, D = dout.shape

        # add node
        dbeta = np.sum(dout, axis=0)
        dgammax = dout
        # multiply node2
        dgamma = np.sum(dgammax * self.xhat, axis=0)
        dxhat = dgammax * self.gamma
        # multiply node1
        divar = np.sum(dxhat * self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar
        # divide node
        dsqrtvar = -1. / (self.sqrtvar ** 2) * divar
        # sqrt(x + eps) node
        dvar = 0.5 * 1. / np.sqrt(self.var + self.eps) * dsqrtvar
        # column mean node2
        dsq = 1. / N * np.ones((N, D)) * dvar
        # square node
        dxmu2 = 2 * self.xmu * dsq
        # minus node
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        # column mean node1
        dx2 = 1. / N * np.ones((N, D)) * dmu
        # input node
        dx = dx1 + dx2

        # memorize
        self.dbeta = dbeta
        self.dgamma = dgamma

        return dx, dgamma, dbeta
