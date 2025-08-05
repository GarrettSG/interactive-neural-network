import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (batch, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define a CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train model
model = create_model()
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.save('guess.keras')

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Reload model
model = tf.keras.models.load_model('guess.keras')

# Predict on custom images
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable")
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)  # invert: black bg, white digit
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_custom_digits(folder="digits"):
    image_number = 0
    while os.path.isfile(f"{folder}/{image_number}.png"):
        path = f"{folder}/{image_number}.png"
        try:
            img = preprocess_image(path)
            prediction = model.predict(img, verbose=0)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction)
            print(f"{path}: {predicted_label} (confidence: {confidence:.2f})")
            plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
            plt.title(f"Prediction: {predicted_label}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error with {path}: {e}")
        image_number += 1

predict_custom_digits()
