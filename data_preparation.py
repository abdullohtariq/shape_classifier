import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    images = []
    labels = []
    label_map = {'circle': 0, 'square': 1, 'triangle': 2}

    for shape in label_map.keys():
        shape_dir = os.path.join(data_dir, shape)
        for img_name in os.listdir(shape_dir):
            img_path = os.path.join(shape_dir, img_name)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 pixels
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label_map[shape])

    images = np.array(images)
    labels = np.array(labels)

    # Normalize the images
    images = images / 255.0

    # Add a channel dimension
    images = np.expand_dims(images, axis=-1)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=3)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
