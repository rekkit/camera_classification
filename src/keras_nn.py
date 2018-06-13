import os
import pandas as pd
from src.utilities import create_train_set, create_validation_set, read_img, dataGenerator

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

# create train and validation sets
train_file_paths = pd.read_csv("./data/train_files.csv")
# create_train_set(train_file_paths, n_crops=1, flip_original=False, folder_name="train_augmented_1")

validation_file_paths = pd.read_csv("./data/validation_files.csv")
# create_validation_set(validation_file_paths, flip_original=False, folder_name="validation_1")

# read the training_data
train_paths = []
y_train = []
train_path = os.getcwd().replace("\\", "/") + "/data/train_augmented_t/"

for k, folder in enumerate(os.listdir(train_path)):
        for file_name in os.listdir(train_path + folder):
                train_paths.append(
                        train_path + "%s/%s" % (folder, file_name)
                )

                y_train.append(k)

# read the validation data
val_paths = []
y_val = []
val_path = os.getcwd() + "/data/validation_t/"

for k, folder in enumerate(os.listdir(val_path)):
        for file_name in os.listdir(val_path + folder):
                val_paths.append(
                        val_path + "%s/%s" % (folder, file_name)
                )

                y_val.append(k)

# create pandas series before passing on
train_paths = pd.Series(train_paths)
val_paths = pd.Series(val_paths)

# data parameters
params = {
    'dim': (512, 512),
    'batch_size': 20,
    'n_classes': 10,
    'n_channels': 3,
    'shuffle': True
}

# create the data generators
training_generator = dataGenerator(train_paths, y_train, **params)
validation_generator = dataGenerator(val_paths, y_val, **params)

# create the NN
# the model will be a sequence of layers
model = Sequential()

# make the CNN
# model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(input_shape=(512, 512, 3), filters=1, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=8, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=8, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# gives us back a <keras.callbacks.History object at 0x112e61a90>
# r = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=15, batch_size=10)
r = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=len(train_paths) // params["batch_size"],
    validation_data=validation_generator,
    validation_steps=len(val_paths) // params["batch_size"],
    use_multiprocessing=False,
    #workers=4
)

print("Returned:", r)



