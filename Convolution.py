import numpy as np


def im2col(input_data, h_f, w_f, stride=1, padding=0):
    """convert 4D tensor into 2D matrix.

    Args:
        input_data (ndarray): (n_batch, channel, height, width) 4D array
        h_f (int): height of filter
        w_f (int): width of filter
        stride (int, optional): stride. Defaults to 1.
        padding (int, optional): padding. Defaults to 0.

    Returns:
        col (ndarray): 2D array.
    """
    n_batch, channel, height, width = input_data.shape
    h_out = (height + 2 * padding - h_f) // stride + 1
    w_out = (width + 2 * padding - w_f) // stride + 1

    mat = np.pad(input_data, [(0, 0), (0, 0),
                              (padding, padding), (padding, padding)],
                 "constant")
    col = np.zeros((n_batch, channel, h_f, w_f, h_out, w_out))

    for h in range(h_f):
        h_max = h + stride * h_out
        for w in range(w_f):
            w_max = w + stride * w_out
            col[:, :, h, w, :, :] = mat[:, :, h:h_max:stride, w:w_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3)
    # (n_batch, h_out, w_out, channel, h_f, w_F)
    col = col.reshape(n_batch * h_out * w_out, -1)
    return col


def col2im(col, input_shape, h_f, w_f, stride=1, padding=0):
    """convert 2Dmatrix into 4D tensor

    Args:
        col (ndarray): data to be converted.
        input_shape (tuple): the shape of input data.
        h_f (int): height of filter
        w_f (int): widht of filter
        stride (int, optional): . Defaults to 1.
        padding (int, optional): . Defaults to 0.

    Returns:
        img(ndarray): 4D array.
    """
    n_batch, channel, height, width = input_shape
    h_out = (height + 2 * padding - h_f) // stride + 1
    w_out = (width + 2 * padding - w_f) // stride + 1
    col = col.reshape(n_batch, h_out, w_out, channel, h_f, w_f)
    # col : (n_batch, channel, h_f, w_f, h_out, w_out)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n_batch, channel, height + 2 * padding + stride - 1,
                    width + 2 * padding + stride - 1))
    for h in range(h_f):
        h_max = h + stride * h_out
        for w in range(w_f):
            w_max = w + stride * w_out
            img[:, :, h:h_max:stride, w:w_max:stride, :, :] += \
                col[:, :, h, w, :, :]

    return img[:, :, padding: padding + height, padding: padding + width]


class Convolution:

    def __init__(self, W, b, stride=1, padding=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding

        # memo
        self.x = None
        self.col = None
        self.col_W = None

        # grad
        self.dW = None
        self.db = None

    def forward(self, x):
        n_f, channel, h_f, w_f = self.W.shape
        n_batch, channel, height, width = x.shape
        h_out = (height + 2 * self.padding - h_f) // self.stride + 1
        w_out = (width + 2 * self.padding - w_f) // self.stride + 1

        # (n_batch * h_out * w_out, channel * h_f * w_f)
        col = im2col(x, h_f, w_f, self.stride, self.padding)
        # (channel * h_f * w_f, n_f)
        col_W = self.W.reshape(n_f, -1).T

        out = np.dot(col, col_W) + self.b  # 積和演算
        # (n_batch, channel, h_out, w_out)
        out = out.reshape(n_batch, h_out, w_out, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        n_f, channel, h_f, w_f = self.W.shape
        # (channel * h_f * w_f, n_f)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, n_f)

        self.db = np.sum(dout, axis=0)  # (n_f,)
        # col.T : (channel * h_f * w_f, n_batch * h_out * w_out)
        # dout : (channel * h_f * w_f, n_f)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(n_f, channel, h_f, w_f)

        # col_W.T : (n_f, channel * h_f * w_f)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, h_f, w_f, self.stride, self.padding)

        return dx
