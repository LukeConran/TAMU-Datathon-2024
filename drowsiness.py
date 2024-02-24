import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define constants
IMAGE_SIZE = (227, 227)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 10

# Define data directories
drowsy_dir = 'drowsy'
non_drowsy_dir = 'non_drowsy'

# Count the number of images in each directory
num_drowsy_images = len(os.listdir(drowsy_dir))
num_non_drowsy_images = len(os.listdir(non_drowsy_dir))

# Load images using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    directory='./',
    classes=['drowsy', 'non_drowsy'],
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=EPOCHS)

# Save the model
model.save('drowsiness_model.h5')

# Example usage:
# Load the trained model
# loaded_model = tf.keras.models.load_model('drowsiness_model.h5')

# Make predictions on new images
# test_image = 'test_image.jpg'
# img = tf.keras.preprocessing.image.load_img(test_image, target_size=IMAGE_SIZE)
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# prediction = loaded_model.predict(img_array)
# print(prediction)