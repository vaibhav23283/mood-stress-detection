import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -----------------------------
# STEP 1: Load and preprocess data
# -----------------------------
train_dir = "dataset/train"
test_dir = "dataset/test"

# Rescale images (0-255 → 0-1)
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),       # all images resized to 48x48 pixels
    batch_size=32,
    color_mode="grayscale",     # dataset is grayscale
    class_mode="categorical"    # multiple categories (emotions)
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# -----------------------------
# STEP 2: Build CNN Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 emotions
])

# -----------------------------
# STEP 3: Compile the model
# -----------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# STEP 4: Train the model
# -----------------------------
model.fit(train_data, validation_data=test_data, epochs=25)

# -----------------------------
# STEP 5: Save the model
# -----------------------------
model.save("emotion_model.h5")
print("✅ Model training complete! Saved as emotion_model.h5")
