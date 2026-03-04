# 6009CMD Topic 1: Image Classification with Deep Convolutional Neural Networks
# Part A – Human-Guided Design (NO Generative AI used)

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

print("=== TOPIC 1 Part A: Model Design & Training – Human-Guided Only ===")

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

print("Data preprocessed (normalised to [0,1])")

# === Model Architecture (designed by me from lectures) ===
# Justification (copy to report):
# - 3 Conv2D blocks with increasing filters (extracts edges/textures like in medical X-rays)
# - MaxPooling reduces size
# - Dropout(0.5) prevents overfitting
# - Softmax gives probability scores for ambiguity detection later

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
es = EarlyStopping(patience=5, restore_best_weights=True)
mc = ModelCheckpoint('models/partA_model.h5', save_best_only=True, monitor='val_accuracy')

history = model.fit(x_train, y_train,
                    epochs=30,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[es, mc])

print("✅ Training finished! Model saved as models/partA_model.h5")

# Save training plot for report
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.savefig('reports/figures/training_history.png')
plt.show()
