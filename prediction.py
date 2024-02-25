import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

if len(sys.argv) == 3:
    loaded_model = tf.keras.models.load_model(sys.argv[1])
    image_name = sys.argv[2]

    img = tf.io.read_file(image_name)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [227, 227])
    resized_img = img.numpy()

processed_image = resized_img / 227.0

processed_image = np.expand_dims(processed_image, axis=0)
prediction = loaded_model.predict(processed_image)

print(prediction)

if (prediction[0][0]) > 0.5:     # this needs to change
    print("drowsy")
else:
    print("not drowsy")
