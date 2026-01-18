import tensorflow as tf
from tensorflow.keras import layers,models


#  DataSet Loading
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Model Creation
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model Compilation

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=5)


# Test Model
model.evaluate(X_test, y_test)

# Calculating loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
