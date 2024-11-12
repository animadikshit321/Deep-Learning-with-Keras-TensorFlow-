# Deep-Learning-with-Keras-TensorFlow-
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
# Load the Stanford Dogs dataset
dataset, info = tfds.load('stanford_dogs', with_info=True, as_supervised=True)

train_data, test_data = dataset['train'], dataset['test']
def preprocess_image(image, label):
    # Resize image to 224x224
    image = tf.image.resize(image, [224, 224])
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    return image, label

# Augmentation function
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Apply preprocessing and augmentation
train_data = train_data.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
# Load a pre-trained model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(120, activation='softmax')  # 120 dog breeds
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              # Train the model
history = model.fit(train_data, 
                    validation_data=test_data, 
                    epochs=10, 
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
                    # Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy:.2f}")

# Predictions
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=np.arange(120))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(120))
disp.plot(cmap=plt.cm.Blues)
plt.show()

