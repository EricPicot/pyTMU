import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def AlexNet(image_width = 480, image_height = 270, channels = 1, NUM_CLASSES = 9):
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=(image_height, image_width, channels)),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="valid"),
        tf.keras.layers.BatchNormalization(),
        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 7
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 8
        tf.keras.layers.Dense(units=NUM_CLASSES,
                              activation=tf.keras.activations.softmax)
    ])

    return model

def digit_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 20, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model
