import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras
from sklearn.model_selection import train_test_split

EPOCHS = 1
IMG_WIDTH = 227 # images are 227 pixels by 227 pixels
IMG_HEIGHT = 227
NUM_CATEGORIES = 2
TEST_SIZE = 0.4
BATCH_SIZE = 64


def main():

    ### FIX ME
    # -------------------------------------------------
    # Check command-line arguments
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    # --------------------------------------------------

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, shuffle=True, random_state=42,
    )

    # Get a compiled neural network
    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    '''
    checkpoint_filepath = '\checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback])
    # model.load_weights(checkpoint_filepath)

    
    STEPS_PER_EPOCH = y_train.size // BATCH_SIZE
    SAVE_PERIOD = 1
    checkpoint_path = "./checkpoints"
    # Create a callback that saves the model's weights every 10 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=int(SAVE_PERIOD * STEPS_PER_EPOCH))
    
    # Train the model with the new callback
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              steps_per_epoch=STEPS_PER_EPOCH,
              epochs=50,
              callbacks=[cp_callback],
              validation_data=(x_train, y_train),
              verbose=0)
    '''

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # initialize empty image and label lists for returning
    images = []
    labels = []

    # 1 = drowsy
    # 0 = nonDrowsy
    # read data from each folder
    # add every image from folder to list
    for i, file in enumerate(os.listdir(os.path.join(data_dir, "drowsy"))):
        if not (i % 2 == 0): continue
        images.append(cv2.resize(cv2.imread(f'{os.path.join(data_dir, "drowsy", file)}', 1), (IMG_WIDTH, IMG_HEIGHT)))
        labels.append(1)
        print("Loading Drowsy: ", i)

    for i, file in enumerate(os.listdir(os.path.join(data_dir, "nonDrowsy"))):
        if not (i % 2 == 0): continue
        images.append(cv2.resize(cv2.imread(f'{os.path.join(data_dir, "nonDrowsy", file)}', 1), (IMG_WIDTH, IMG_HEIGHT)))
        labels.append(0)
        print("Loading nonDrowsy: ", i)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # check for provided model

    try:
        if sys.argv[3]:
            new_model = tf.keras.models.load_model(sys.argv[3])
            return new_model
    except:
        pass

    model = tf.keras.models.Sequential([

        # add a convolutional layer with 32 filters an a 3x3 kernal
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # add a max-pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # flatten data
        tf.keras.layers.Flatten(),

        # add a hidden layer with relu activation and dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # add an output later with output units for each category
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
