import os
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle
from src.utilities import create_train_set, create_validation_set, transform_images, read_img
from src.dl_layers import hiddenLayer, convolutionalLayer, convPoolLayer

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

    def fit(self, x_paths, y, x_val_paths, y_val, img_w, img_h, n_channels, session, n_epochs, batch_size=20, print_step=20, show_fig=True,
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
        self.costs_t = []
        self.accuracy_t = []
        self.costs_v = []
        self.accuracy_v = []

        # set the number of steps per epoch
        n_steps = len(x_paths) // batch_size

        for i in range(n_epochs):
            x_paths, y = shuffle(x_paths, y)

            for j in range(n_steps):
                x_batch_paths = x_paths[j*batch_size: (j+1)*batch_size]
                y_batch = y[j*batch_size: (j+1)*batch_size]

                # create x_batch
                x_batch = []
                for file_path in x_batch_paths:
                    x_batch.append(read_img(file_path))

                x_batch = np.reshape(x_batch, [batch_size, img_w, img_h, -1])

                # perform cost minimization step
                self.session.run(
                    self.train_op,
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                if j % print_step == 0 and j > 0:
                    # because of memory issues we have to calculate the validation cost / accuracy batch by batch
                    total_cost = 0
                    total_correct = 0
                    x_val_paths, y_val = shuffle(x_val_paths, y_val)

                    for k in range(1000 // batch_size):
                        # create x_batch
                        x_batch_val = []
                        for file_path in x_val_paths[k*batch_size: (k+1)*batch_size]:
                            x_batch_val.append(read_img(file_path))

                        x_batch_val = np.reshape(x_batch_val, [batch_size, img_w, img_h, -1])
                        y_batch_val = y_val[k*batch_size: (k+1)*batch_size]

                        # run training step and get the cost for the particular batch
                        batch_cost, predictions = self.session.run(
                            (self.cost, self.predict_op),
                            feed_dict={self.tfX: x_batch_val, self.tfT: y_batch_val}
                        )

                        total_cost += batch_cost
                        total_correct += np.sum(predictions == y_batch_val)

                    # add the cost and accuracy to the corresponding lists
                    self.costs_v.append(total_cost / 1000)
                    self.accuracy_v.append(total_correct / 1000)

                    # add the cost for the training data batch
                    batch_cost, predictions = self.session.run(
                        (self.cost, self.predict_op),
                        feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                    )

                    self.costs_t.append(batch_cost)
                    self.accuracy_t.append(np.mean(y_batch == predictions))

                    print(
                        "Epoch: %d." % i,
                        "Step %d of %d completed." % (j, n_steps),
                        "Batch cost: %.2f." % self.costs_t[-1],
                        "Batch accuracy: %.2f." % self.accuracy_t[-1],
                        "Validation cost: %.2f." % self.costs_v[-1],
                        "Validation accuracy: %.2f." % self.accuracy_v[-1]
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
train_file_paths = pd.read_csv("./data/train_files.csv")
train_file_paths["folder_name"] = train_file_paths.folder_path.apply(lambda x: x.split("/")[-2])
train_file_paths = train_file_paths.loc[
    ~np.in1d(train_file_paths.folder_name, ["HTC-1-M7", "iPhone-4s", "iPhone-6"])
].reset_index(drop=True)

create_train_set(train_file_paths=train_file_paths, n_crops=3, folder_name="train_aug_1")

# read the training_data
x_paths = []
y_train = []
train_path = os.getcwd().replace("\\", "/") + "/data/train_aug/"

for k, folder in enumerate(os.listdir(train_path)):
    for file_name in os.listdir(train_path + folder):
        x_paths.append(
            train_path + "%s/%s" % (folder, file_name)
        )

        y_train.append(k)

# read the validation data
val_paths = []
y_val = []
val_path = os.getcwd() + "/data/validation/"

for k, folder in enumerate(os.listdir(val_path)):
    for file_name in os.listdir(val_path + folder):
        val_paths.append(
            val_path + "%s/%s" % (folder, file_name)
        )

        y_val.append(k)


# define the NN
conv_net = convolutionalNeuralNetwork(
    layers=[
        convolutionalLayer(3, 3, 64),
        convPoolLayer(3, 3, 64, [1, 2, 2, 1], [1, 2, 2, 1]),
        convolutionalLayer(3, 3, 32),
        convPoolLayer(3, 3, 32, [1, 2, 2, 1], [1, 2, 2, 1]),
        convPoolLayer(3, 3, 16, [1, 2, 2, 1], [1, 2, 2, 1]),
        hiddenLayer(10)
    ],
    activation_fn=tf.nn.relu
)

# fit the NN
conv_net.fit(
    x_paths=x_paths, y=y_train, x_val_paths=val_paths, y_val=y_val, img_w=512, img_h=512, n_channels=3, session=tf.Session(),
    n_epochs=10, batch_size=20
)

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
