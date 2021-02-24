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


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.max_idx = None

    def forward(self, x):
        n_batch, channel, height, width = x.shape
        out_h = (height + 2 * self.padding - self.pool_h) // self.stride + 1
        out_w = (width + 2 * self.padding - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w,
                     stride=self.stride, padding=self.padding)

        # col : (n_batch * h_out * w_out * channel, pool_h * pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        max_idx = np.argmax(col, axis=1)
        out = np.max(
            col,
            axis=1).reshape(
            n_batch,
            out_h,
            out_w,
            channel).transpose(
            0,
            3,
            1,
            2)
        # out: (n_batch, channel, out_h, out_w)
        self.x = x
        self.max_idx = max_idx

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        # duot : (n_batch, out_h, out_w, channel)
        pool_size = self.pool_h * self.pool_w
        # dmax : (n_batch*out_h*out_w*channel, pool_size)
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.max_idx.size), self.max_idx.flatten()] =\
            dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        # dmax : (n_batch, out_h, out_w, channel, pool_size)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # dcol: (n_batch * out_h * out_w, channel * pool_size) (2D)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride,
                    self.padding)

        return dx
