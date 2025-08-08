import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras import layers, models

# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def create_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2d_1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train model
print("Training model...")
model = create_model()
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Export to TensorFlow.js - save to Next.js public directory
print("Exporting model to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, './neural-network-app/public/model')

# Save sample test data to public directory
print("Saving sample test data...")
# Convert numpy arrays to JSON for easier loading in browser
test_sample = {
    'images': x_test[:50].tolist(),  # Convert first 50 images to list
    'labels': y_test[:50].tolist()   # Convert corresponding labels to list
}

import json
with open('./neural-network-app/public/test_data.json', 'w') as f:
    json.dump(test_sample, f)

print("Export complete!")