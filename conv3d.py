import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
from lasagne.theano_extensions import conv, padding
from lasagne.layers.base import Layer
from lasagne.layers import conv
import theano

__all__ = [
    "Conv3DLayer",
]

class Conv3DLayer(Layer):
    """
    lasagne.layers.Conv3DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify,
    convolution=theano.sandbox.cuda.dnn.dnn_conv3d, **kwargs)

    3D convolutional layer

    Performs a 3D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 3-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'valid'`` is only considered. others are under construction

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        4D tensor.

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. This should initialize the
        layer weights to a 5D array with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns, filter depth)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If None is provided, the
        layer will have no biases. This should initialize the layer biases to
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, input_rows, input_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable
        Variable representing the filter weights.

    b : Theano shared variable
        Variable representing the biases.

    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=theano.sandbox.cuda.dnn.dnn_conv3d, **kwargs):
        super(Conv3DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 3)
        self.stride = as_tuple(stride, 3)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')

        if pad == 'valid':
            self.pad = (0, 0, 0)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad,3,int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3], self.output_shape[4])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1], self.filter_size[2])

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 3

        output_rows = conv.conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         pad[0])

        output_columns = conv.conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            pad[1])

        output_depth = conv.conv_output_length(input_shape[4],
                                            self.filter_size[2],
                                            self.stride[2],
                                            pad[2])

        return (input_shape[0], self.num_filters, output_rows, output_columns, output_depth)

    def get_output_for(self, input, input_shape=None, **kwargs):
        if self.stride == (1, 1, 1) and self.pad == 'same':
            conved = self.convolution(img=input, 
                                      kerns=self.W, 
                                      border_mode='full',
                                      subsample=self.stride,
                                      conv_mode='conv')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            shift_z = (self.filter_size[2] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x, 
                                  shift_y:input.shape[3] + shift_y,
                                  shift_z:input.shape[4] + shift_z]
        else:
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        self.filter_size[0] // 2),
                       (self.filter_size[1] // 2,
                        self.filter_size[1] // 2),
                       (self.filter_size[2] // 2,
                        self.filter_size[2] // 2)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (self.pad[2], self.pad[2])]
            if pad != [(0, 0), (0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=3)
                input_shape = (input_shape[0], input_shape[1], 
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1],
                               None if input_shape[4] is None else
                               input_shape[4] + pad[2][0] + pad[2][1])

            conved = self.convolution(img=input, 
                                  kerns=self.W, 
                                  border_mode=border_mode,
                                  subsample=self.stride,
                                  conv_mode='conv')

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2, 3)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x', 'x')

        return self.nonlinearity(activation)