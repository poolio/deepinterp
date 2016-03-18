import numpy as np
import lasagne
import theano
import theano.tensor as T
import os

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from theano.gradient import consider_constant, disconnected_grad

from lasagne.layers.special import _linspace, _meshgrid
from lasagne.layers.special import _interpolate as _interpolate_bilinear

def get_translation_params():
    theta_indices = [1] # 2
    ntheta = len(theta_indices)
    theta = theano.tensor.fmatrix('translation')
    batch_size = 1
    loc_params_init = np.zeros((batch_size, 6))
    loc_params_init[:, 0] = 1
    loc_params_init[:, 4] = 1
    loc_params_init = theano.tensor.constant(loc_params_init.astype(np.float32))
    # Note this cannot be symbolic
    loc_params = theano.tensor.set_subtensor(loc_params_init[0:1, 2:3] , theta)
    return theta, loc_params

def dg2(x):
    return disconnected_grad(disconnected_grad(x))

def _cubic_conv_weights(x):
    ax = T.abs_(x)# * dg2(T.sgn(x) + T.lt(T.abs_(T.sgn(x)),0.9))#T.abs_(x)
    ax = T.switch(T.eq(ax,0), 0,ax )
    ax3 = ax * ax * ax
    #return ax3
    a = -0.5
    wx = T.switch((ax <= 1.0), (a+2.0) * ax3 - (a+3)*x*x + 1.0, T.zeros_like(x))
    wx = wx + T.switch((ax > 1.0) & (ax < 2.0), a * ax3 - 5 * a * x*x + 8 * a *ax - 4 * a, T.zeros_like(x))
    return wx

def _interpolate_bicubic(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)
    grid = _meshgrid(out_height, out_width)
    x_grid_flat = grid[0].flatten()
    y_grid_flat = grid[1].flatten()

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)
    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    #return T.concatenate(((x0-x).dimshuffle(0, 'x')**2, 0.0*dg2(x.dimshuffle(0, 'x')), 0.0*dg2(x0.dimshuffle(0, 'x'))), 1)

    offsets = np.arange(-1, 3).astype(int)
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    # Need to convert (x, y) to linear
    def _flat_idx(xx, yy, dim2=dim2):
        return base + yy * dim2 + xx
    y_locs = [y0 + offset for offset in offsets]
    ys = [T.clip(loc, 0, height - 1) for loc in y_locs]

    def _cubic_interp_dim(im_flat, other_idx):
        """Cubic interpolation along a dimension
        """
        neighbor_locs = [x0 + offset for offset in offsets]
        neighbor_idx = [T.clip(nloc, 0, width - 1) for nloc in neighbor_locs]
        xidxs = neighbor_idx
        yidxs = [other_idx] * len(neighbor_idx)
        neighbor_idxs = [_flat_idx(xidx, yidx) for xidx, yidx in zip(xidxs, yidxs)]
        values = [im_flat[idx] for idx in neighbor_idxs]
        weights = [_cubic_conv_weights(dg2(nloc) - x).dimshuffle(0, 'x')  for nloc in neighbor_locs]
        # Interpolate along x direction
        out = T.sum([dg2(v) * w for w, v in zip(weights, values)], axis=0) / T.sum(weights, axis=0)
        return out
    im_flat = im.reshape((-1, channels))
    ims = [_cubic_interp_dim(im_flat, yidx) for yidx in ys]
    yweights =  [_cubic_conv_weights(dg2(yloc) - y).dimshuffle(0, 'x') for yloc in y_locs]
    out = T.sum([v *  _cubic_conv_weights(dg2(yloc) - y).dimshuffle(0, 'x') for v, yloc in zip(ims,  y_locs)], axis=0) / T.sum(yweights, axis=0)
    return out

def transform_img(theta, input, downsample_factor, interp='bicubic'):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(height / downsample_factor[0], 'int64')
    out_width = T.cast(width / downsample_factor[1], 'int64')
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    if interp == 'bicubic':
        interpolator = _interpolate_bicubic
    elif interp == 'bilinear':
        interpolator = _interpolate_bilinear
    else:
        raise ValueError("Interpolation must be \"bilinear\" or \"bicubic\", got \"%s\""%interp)
    input_transformed = interpolator(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width)
    output = T.reshape(
       input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output

class CubicSpatialTransformer(lasagne.layers.TransformerLayer):
    #def __init__(self, *args, **kwargs):
    def __init__(self, incoming, localization_network, interp='bicubic', **kwargs):
        super(CubicSpatialTransformer, self).__init__(incoming, localization_network, **kwargs)
        self.interp = interp
    def get_output_for(self, inputs, **kwargs):
        input, theta = inputs
        return T.cast(transform_img(theta, input, self.downsample_factor, interp=self.interp), 'float32')

