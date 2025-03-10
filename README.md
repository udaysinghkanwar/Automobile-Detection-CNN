# Automobile-Detection-CNN
# CIFAR-10 Image Classification Using TensorFlow

## Overview
This project adapts the Fashion MNIST classification code to use the **CIFAR-10 dataset**, a dataset consisting of 60,000 color images in 10 classes, with 6,000 images per class. The project implements a **Convolutional Neural Network (CNN)** for classification, saves and reloads the trained model using **Pickle**, and allows optional user image uploads for real-time classification.

## Features
- **Uses CIFAR-10 dataset** instead of Fashion MNIST.
- **Handles color images (RGB)** instead of grayscale.
- **Trains a CNN model** for image classification.
- **Saves and reloads the trained model** using JSON and Pickle.
- **Allows user-uploaded images** for classification.

## Dependencies
Install TensorFlow and required libraries before running the code:
```bash
pip install tensorflow numpy matplotlib pickle5 pillow
```

## Dataset
We use the **CIFAR-10 dataset**, which consists of 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is loaded directly from TensorFlow:
```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

## Model Architecture
The CNN model consists of:
- **Convolutional layers** (feature extraction)
- **Flatten layer** (to convert features into a 1D vector)
- **Dropout layers** (to prevent overfitting)
- **Dense layers** (fully connected layers)
- **Softmax output layer** (for classification)

```python
model = Model(i, x)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Training the Model
The model is trained using `fit()` on CIFAR-10 training data:
```python
r = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5
)
```

## Saving and Reloading the Model
Instead of saving using `model.save()`, we pickle the model architecture and weights separately:

### Saving the Model
```python
model_json = model.to_json()
model_weights = model.get_weights()
with open("cifar10_model.pkl", "wb") as f:
    pickle.dump((model_json, model_weights), f)
```

### Reloading the Model
```python
with open("cifar10_model.pkl", "rb") as f:
    loaded_model_json, loaded_model_weights = pickle.load(f)
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.set_weights(loaded_model_weights)
```

## Making Predictions
A random test image is used for verification:
```python
idx = random.randint(0, len(x_test) - 1)
test_image = x_test[idx]
true_label = y_test[idx]

pred_probs = loaded_model.predict(test_image[np.newaxis, ...])
pred_label = np.argmax(pred_probs)
```

## User Image Upload (Optional)
Users can upload their own images for classification:
```python
from google.colab import files
from PIL import Image

uploaded = files.upload()
img_name = list(uploaded.keys())[0]
img = Image.open(img_name).resize((32, 32))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)
pred_probs = loaded_model.predict(img)
pred_label = np.argmax(pred_probs)
```

## How to Run
1. Install dependencies.
2. Run the script to train the model.
3. Save and reload the model using Pickle.
4. Use the model to classify images from the dataset or user-uploaded images.

