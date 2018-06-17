import keras
import pandas as pd
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet50 import ResNet50
# from keras.applications.densenet import DenseNet201
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from src.utilities import create_data_generator, train_model, plot_training_perf, create_submission

# tensor
input_tensor = Input(shape=(512, 512, 3))

# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer and dropout
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
# and a logistic layer
predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all the layers of the base model (or however many you want)
for layer in base_model.layers:
    layer.trainable = False

# Adam optimizer
optimizer = Adam()  # Adam(lr=0.0001)  #
# optimizer = RMSprop(lr=0.0001)

# compile the model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# data parameters
params = {
    "target_size": (512, 512),
    "batch_size": 64,
    "class_mode": "categorical",
    "shuffle": True,
    "rescale": 1 / 255
}

# create the data generators
training_generator = create_data_generator(
    data_directory="./data/train/",
    **params
)

validation_generator = create_data_generator(
    data_directory="./data/validation/",
    **params
)

# model number
model_n = 8  # change this depending on the number of the model you're training
model_name = "model_%d" % model_n

# train the model and save the metrics
r = train_model(
    model,
    training_generator,
    validation_generator,
    params,
    epochs=3,
    callbacks=[
            keras.callbacks.ModelCheckpoint(filepath="./models/%s/incv3_{epoch:02d}_{val_loss:.2f}.hdf5" % model_name)
        ],
    model_n=model_n
)

# read the metrics generated
dt = pd.read_csv("./models/%s/metrics.csv" % model_name)

# plot the model's performance
plot_training_perf(dt, "accuracy", model_name)
plot_training_perf(dt, "loss", model_name)

# load model - uncomment if you want to choose your model, otherwise use the model just trained.
model_n = 6
model_name = "model_%d" % model_n
model = keras.models.load_model("./models/%s/incv3_03_1.88.hdf5" % model_name)

# create submission
create_submission(model, training_generator, model_n)
