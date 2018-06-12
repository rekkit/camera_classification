import numpy as np
import tensorflow as tf

class hiddenLayer(object):
    with tf.variable_scope("hidden_layer"):
        def __init__(self, n_out):
            self.n_out = n_out

            # these are the variables that will be assigned a value when the graph of the NN is being built
            self.n_in = None
            self.layer_id = None
            self.activation_fn = None
            self.w = None
            self.b = None

        def initialize(self, n_in, layer_id, activation_fn):
            # assign values to the rest of the class variables
            self.n_in = n_in
            self.layer_id = layer_id
            self.activation_fn = activation_fn

            # create initializer and initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.w = tf.Variable(
                initializer((n_in, self.n_out)),
                name="w_%d" % self.layer_id
            )

            self.b = tf.Variable(
                np.zeros(self.n_out, dtype=np.float32),
                name="b_%d" % self.layer_id
            )

        def forwardLogits(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after multiplying the input by the weights and adding the biases. These are the values
                     that are fed into the activation function.
            """
            return tf.matmul(x, self.w) + self.b

        def forward(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after performing forward propagation in this layer.
            """
            return self.activation_fn(
                self.forwardLogits(x)
            )

class convolutionalLayer(object):
    with tf.variable_scope("convolutional_layer"):
        def __init__(self, filter_w, filter_h, fm_out):
            self.filter_w = filter_w
            self.filter_h = filter_h
            self.fm_out = fm_out

            # these are the variables that will be assigned a value when the graph of the NN is being built
            self.fm_in = None
            self.layer_id = None
            self.w = None
            self.b = None
            self.activation_fn = None

        def initialize(self, fm_in, layer_id, activation_fn):
            self.fm_in = fm_in
            self.layer_id = layer_id
            self.activation_fn = activation_fn

            # create initializer and initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.w = tf.Variable(
                initializer((self.filter_w, self.filter_h, self.fm_in, self.fm_out)),
                name="w_%d" % self.layer_id
            )

            self.b = tf.Variable(
                np.zeros(shape=self.fm_out, dtype=np.float32),
                name="b_%d" % self.layer_id
            )

        def forwardLogits(self, x):
            return tf.nn.conv2d(x, self.w, strides=[1, 1, 1, 1], padding="SAME") + self.b

        def forward(self, x):
            return self.activation_fn(
                self.forwardLogits(x)
            )

        def output_size(self, img_w, img_h):
            return img_w, img_h

class convPoolLayer(convolutionalLayer):
    with tf.variable_scope("conv_pool_layer"):
        def __init__(self, filter_w, filter_h, fm_out, ksize, strides):
            convolutionalLayer.__init__(self, filter_w, filter_h, fm_out)

            # initialize the additional attributes
            self.ksize = ksize
            self.strides = strides

        def forwardLogits(self, x):
            return tf.nn.max_pool(
                tf.nn.conv2d(x, self.w, strides=[1, 1, 1, 1], padding="SAME") + self.b,
                ksize=self.ksize,
                strides=self.strides,
                padding="VALID"
            )

        def output_size(self, img_w, img_h):
            return int((img_w - self.ksize[1]) / self.strides[1] + 1), int((img_h - self.ksize[2]) / self.strides[2] + 1)
