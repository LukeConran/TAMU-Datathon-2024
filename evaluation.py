import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

if len(sys.argv) == 3:
    loaded_model = tf.keras.models.load_model(sys.argv[1])
    image_name = sys.argv[2]

    # load image via tf.io
    img = tf.io.read_file(image_name)

    # convert to tensor (specify 3 channels explicitly since png files contains additional alpha channel)
    # set the dtypes to align with pytorch for comparison since it will use uint8 by default
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    # (x, y, 3)

    # resize tensor to 227 x 227
    tensor = tf.image.resize(tensor, [227, 227])
    # (227, 227, 3)

    # add another dimension at the front to get NHWC shape
    input_tensor = tf.expand_dims(tensor, axis=0)
    # Interpret the evaluation metrics
    accuracy = loaded_model.evaluate(input_tensor)
    print(accuracy)
    # if accuracy >= 0.5:  # adjust the threshold as needed
    #     print("Image fits the criteria")
    # else:
    #     print("Image does not fit the criteria")
