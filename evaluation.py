import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

if len(sys.argv) == 3:
    loaded_model = tf.keras.models.load_model(sys.argv[1])
    image_name = sys.argv[2]

    model = tf.keras.models.load_model(sys.argv[1])

    image = tf.keras.preprocessing.image.load_img(image_name, target_size=(227, 227))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 227

    image = tf.expand_dims(image, axis=0)

    # Prepare the label (assuming binary classification, adjust as needed)
    # 1 if the image fits the criteria, 0 otherwise
    label = 1  # or 0 depending on your criteria

    # Evaluate the model
    loss, accuracy = loaded_model.evaluate(image, [label])

    # Interpret the evaluation metrics
    print(accuracy)
    if accuracy >= 0.5:  # adjust the threshold as needed
        print("Image fits the criteria")
    else:
        print("Image does not fit the criteria")
