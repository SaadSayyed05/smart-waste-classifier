# Smart Waste Classifier - Training Pipeline
# FINAL SCRIPT using MobileNetV2.
# This version corrects the fine-tuning epoch calculation and the final prediction steps.

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- 1. Constants and Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
NUM_EPOCHS_HEAD = 15
NUM_EPOCHS_FINE_TUNE = 10
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
MODEL_SAVE_PATH = 'waste_classifier_model.h5'

# --- 2. Data Preprocessing and Augmentation ---

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 3. Model Building (Transfer Learning with MobileNetV2) ---

input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_tensor=input_tensor)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Phase 1: Train the Classifier Head ---

print("--- Starting Phase 1: Training the model head (using MobileNetV2) ---")

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_head = model.fit(
    train_generator,
    epochs=NUM_EPOCHS_HEAD,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Phase 2: Fine-Tuning (Optional) ---

print("\n--- Starting Phase 2: Fine-tuning the model ---")

for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

total_epochs = NUM_EPOCHS_HEAD + NUM_EPOCHS_FINE_TUNE

history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history_head.epoch[-1],
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 6. Model Evaluation and Visualization ---

print("\n--- Evaluating the final model ---")

acc = history_head.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history_head.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history_head.history['loss'] + history_fine_tune.history['loss']
val_loss = history_head.history['val_loss'] + history_fine_tune.history['val_loss']

if not os.path.exists('evaluation_plots'):
    os.makedirs('evaluation_plots')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.axvline(x=len(history_head.history['accuracy']) - 1, color='r', linestyle='--')
plt.savefig('evaluation_plots/accuracy_plot.png')

plt.subplot(1, 2, 2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.axvline(x=len(history_head.history['loss']) - 1, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('evaluation_plots/loss_plot.png')
plt.show()

validation_generator.reset()
# --- FIX: Convert the result of np.ceil to an integer ---
steps_for_prediction = int(np.ceil(validation_generator.samples / BATCH_SIZE))
Y_pred = model.predict(validation_generator, steps=steps_for_prediction)
y_pred = (Y_pred > 0.5).astype(int).flatten()

y_true = validation_generator.classes

print('\nConfusion Matrix')
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=validation_generator.class_indices.keys(),
            yticklabels=validation_generator.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('evaluation_plots/confusion_matrix.png')
plt.show()

print('\nClassification Report')
print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))

print(f"\n--- Saving the trained model to {MODEL_SAVE_PATH} ---")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")

def plot_sample_predictions(model, generator):
    generator.reset()
    x_batch, y_batch_true = next(generator)
    y_batch_pred_probs = model.predict(x_batch)
    y_batch_pred = (y_batch_pred_probs > 0.5).astype(int).flatten()
    class_labels = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(15, 10))
    for i in range(min(9, len(x_batch))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i])
        true_label = class_labels[y_batch_true[i]]
        pred_label = class_labels[y_batch_pred[i]]
        title_color = 'g' if true_label == pred_label else 'r'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('evaluation_plots/sample_predictions.png')
    plt.show()

print("\n--- Visualizing sample predictions ---")
plot_sample_predictions(model, validation_generator)
