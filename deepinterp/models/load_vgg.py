import numpy as np
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from collections import OrderedDict
import theano
no_pool = True
from lasagne.nonlinearities import elu
nonlinearity = elu
if no_pool:
    from lasagne.layers import Conv2DLayer as ConvLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    pmode = 'average_inc_pad'
    net = OrderedDict()
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, nonlinearity=nonlinearity)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    #net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False, mode=pmode)
    net['pool1'] = net['norm1']
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, stride=3, nonlinearity=nonlinearity)
    net['pool2'] = net['conv2']#PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, stride=2, nonlinearity=nonlinearity)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, nonlinearity=nonlinearity)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, stride=3, nonlinearity=nonlinearity)
    net['pool5'] = net['conv5']#PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096, nonlinearity=nonlinearity)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096, nonlinearity=nonlinearity)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=None)#lasagne.nonlinearities.softmax)
    output_layer = net['fc8']
else:
    net = OrderedDict()
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=None)#lasagne.nonlinearities.softmax)
    output_layer = net['fc8']

import pickle

model_file = '/home/poole/deepinterp/deepinterp/models/vgg_cnn_s.pkl'
model = pickle.load(open(model_file))
CLASSES = model['synset words']
MEAN_IMAGE = model['mean image']

lasagne.layers.set_all_param_values(output_layer, model['values'])

import io
import skimage.transform

def prep_image(im):
   # ext = url.split('.')[-1]
    #im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    return rawim, floatX(im[np.newaxis])

def unprep(im):
    return np.clip((im[0] + MEAN_IMAGE)[::-1].transpose(1, 2, 0) / 255.0, 0.0, 1.0)
