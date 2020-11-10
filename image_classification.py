
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def create_fc_model():
  model = keras.Sequential([
      keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(28, 28, 1)),  
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10, activation="softmax")
  ])
  return model
  
def create_cnn_model():
  model = keras.Sequential([
      keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation ='relu', padding='same'),
      keras.layers.MaxPool2D(pool_size=(2,2)),
      keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'),
      keras.layers.MaxPool2D(pool_size=(2,2)),      
      keras.layers.Dropout(0.25),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dropout(0.25),
      keras.layers.Dense(10, activation="softmax")
  ])
  return model

model=create_cnn_model()
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
epochs=10
history = model.fit(train_images, train_labels, batch_size = 100, epochs = epochs, validation_data =(test_images, test_labels), shuffle=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


