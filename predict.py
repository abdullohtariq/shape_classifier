import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Define class names
class_names = ['circle', 'square', 'triangle']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Load the trained model
model = load_model('shape_classifier_model.h5')

# Example prediction
image_path = 'New data/square/square_11.png'
img_array = preprocess_image(image_path)
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction, axis=1)[0]
predicted_class_name = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class_name}")
