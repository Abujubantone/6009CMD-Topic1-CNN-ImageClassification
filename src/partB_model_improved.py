# 6009CMD Topic 1 - Part B (GenAI assisted) - FIXED for TF 2.17
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np

print("=== Part B - GenAI-assisted improved model (fixed version) ===")

# Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
y_train = y_train.flatten()
y_test  = y_test.flatten()

# Data Augmentation (from GenAI)
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True,
                             zoom_range=0.1, fill_mode="reflect")
datagen.fit(x_train)

# Model (deeper + BatchNorm from GenAI)
def build_model():
    inputs = layers.Input(shape=(32,32,3))
    x = inputs
    for filters in [32, 64, 128]:
        for _ in range(2):
            x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.4)(x)          # MC Dropout will be enabled later
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs, outputs)

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Callbacks (from GenAI)
callbacks = [
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("PartB/models/partB_best_model.h5", save_best_only=True, monitor="val_accuracy")
]

# Train
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=50,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

print("✅ Part B training finished!")
print("Model saved in PartB/models/")
