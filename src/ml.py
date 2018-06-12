import os
import sys
# add the subfolder src to the system path
sys.path.append(os.getcwd().replace("\\", "/") + "/src")

# other imports
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle
from utilities import create_train_set, create_validation_set, transform_images, read_img
from dl_layers import hiddenLayer, convolutionalLayer, convPoolLayer

class convolutionalNeuralNetwork(object):
    def __init__(self, layers, activation_fn):
        self.layers = layers
        self.activation_fn = activation_fn
        self.n_classes = self.layers[-1].n_out  # the last layer should be the dimension of the number of classes

    def initializeLayers(self, img_w, img_h, fm_in):
        with tf.variable_scope("initialize_layers"):
            fm_in = fm_in
            for i, layer in enumerate(self.layers[:-1]):
                # add the missing information to the conv / convPool layer
                layer.initialize(fm_in=fm_in, layer_id=i, activation_fn=self.activation_fn)

                # update the image and (input) feature map dimensions
                img_w, img_h = layer.output_size(img_w=img_w, img_h=img_h)
                fm_in = layer.fm_out

            # create a fully connected layer to output class probabilities
            self.layers[-1].initialize(
                n_in=img_w * img_h * self.layers[-2].fm_out,
                layer_id=i+1,
                activation_fn=self.activation_fn
            )

    def forwardHiddenLayers(self, x):
        with tf.name_scope("forward_hidden_layers"):
            z = x
            for layer in self.layers[:-1]:
                z = layer.forward(z)

            return z

    def forwardLogits(self, x):
        with tf.name_scope("forward_logits"):
            z = self.forwardHiddenLayers(x)

            # here we have to reshape z so that we can pass it to the fully connected layer
            shape = tf.shape(z)
            z = tf.reshape(z, [shape[0], -1])

            return self.layers[-1].forwardLogits(z)

    def forwardProbabilities(self, x):
        with tf.name_scope("forward_probabilities"):
            z = self.forwardLogits(x)

            return tf.nn.softmax(z)

    def predict(self, x):
        with tf.name_scope("predict"):
            z = self.forwardProbabilities(x)

            return tf.argmax(z, axis=1)

    def initializePlaceholders(self, img_w, img_h, n_channels):
        with tf.name_scope("initialize_placeholders"):
            self.tfX = tf.placeholder(
                dtype=tf.float32,
                shape=(None, img_w, img_h, n_channels),
                name="tfX"
            )

            self.tfT = tf.placeholder(
                dtype=tf.int32,
                shape=(None, ),
                name="tfT"
            )

    def initializeCost(self, n_classes):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.tfT, n_classes),
                logits=self.forwardLogits(self.tfX)
            )
        )

    def initializeOperations(self):
        # train operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # return probabilities
        self.return_probs_op = self.forwardProbabilities(self.tfX)

        # return predictions
        self.predict_op = self.predict(self.tfX)

    def setSession(self, session):
        self.session = session

    def fit(self, x, y, img_w, img_h, n_channels, session, n_epochs, batch_size=20, print_step=20, show_fig=True,
            output_fig=False):
        # set the session to be used
        self.setSession(session)
        
        # one-hot encode the target variable
        #y = tf.one_hot(y, depth=self.n_classes)

        # initialize layers using the previously calculated value
        self.initializeLayers(img_w=img_w, img_h=img_h, fm_in=n_channels)

        # initialize placeholders, cost function and operations
        self.initializePlaceholders(img_w=img_w, img_h=img_h, n_channels=n_channels)
        self.initializeCost(n_classes=self.n_classes)
        self.initializeOperations()

        # initialize global variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # create lists to store the cost / accuracy values for each epoch
        self.costs = []
        self.accuracy = []

        # set the number of steps per epoch
        n_steps = x.shape[0] // batch_size

        for i in range(n_epochs):
            x, y = shuffle(x, y)

            for j in range(n_steps):
                x_batch = x[j*batch_size: (j+1)*batch_size, ]
                y_batch = y[j*batch_size: (j+1)*batch_size]

                # perform cost minimization step
                self.session.run(
                    self.train_op,
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                if j % print_step == 0 and j > 0:
                    # run training step and get the cost for the particular batch
                    _, batch_cost, predictions = self.session.run(
                        (self.train_op, self.cost, self.predict_op),
                        feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                    )

                    # add the cost and accuracy to the corresponding lists
                    self.costs.append(batch_cost)
                    self.accuracy.append(np.mean(y_batch == predictions))

                    print(
                        "Epoch: %d." % i,
                        "Step %d of %d completed." % (j, x.shape[0]),
                        "Cost: %.2f." % self.costs[-1],
                        "Accuracy: %.2f." % self.accuracy[-1]
                    )

        if show_fig:
            self.plotMetrics(output_fig=output_fig)

    def plotMetrics(self, output_fig=False):
        g1 = go.Scatter(
            x=np.linspace(1, len(self.costs), len(self.costs)),
            y=self.costs,
            name="Cost"
        )

        g2 = go.Scatter(
            x=np.linspace(1, len(self.costs), len(self.costs)),
            y=self.accuracy,
            name="Accuracy"
        )

        figure = plotly.tools.make_subplots(2, 1, True, print_grid=False, subplot_titles=("Cost", "Accuracy"))
        figure.append_trace(g1, 1, 1)
        figure.append_trace(g2, 2, 1)

        if output_fig:
            plotly.offline.plot(
                figure,
                image="png",
                image_filename="poetry_generator_rnn"
             )

        else:
            plotly.offline.plot(figure)

# create the validation set
# create_validation_set(n_validation_files=75, folder_name="validation_1")

# create train set
# create_train_set(n_crops=3, folder_name="train_aug_1")

# read the training_data
train = []
train_y = []
data_path = "C:/Repos/camera_classification/data/train_aug/"

for k, folder in enumerate(os.listdir(data_path)):
    for file_name in os.listdir(data_path + folder):
        if "_unalt_0" in file_name:
            train.append(
                read_img(data_path + "%s/%s" % (folder, file_name)) / 255
            )

            train_y.append(k)

# create matrix from data
dt_train = np.reshape(train, [len(train), 512, 512, 3])
del train

# define the NN
conv_net = convolutionalNeuralNetwork(
    layers=[
        convolutionalLayer(7, 7, 64),
        convPoolLayer(7, 7, 32, [1, 2, 2, 1], [1, 2, 2, 1]),
        hiddenLayer(10)
    ],
    activation_fn=tf.nn.relu
)

# fit the NN
conv_net.fit(x=dt_train, y=train_y, img_w=512, img_h=512, n_channels=3, session=tf.Session(), n_epochs=10, batch_size=20)

# conv_net.session.run(
#     tf.shape(
#         conv_net.layers[1].forward(
#             conv_net.layers[0].forward(conv_net.tfX)
#         )
#     ),
#     feed_dict={conv_net.tfX: dt_train[:20, ]}
# )
#
# conv_net.session.run(
#     conv_net.layers[1].output_size(img_w=512, img_h=512),
#     feed_dict={conv_net.tfX: dt_train[:20, ]}
# )
