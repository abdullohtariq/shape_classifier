# Shape Classifier

## Video Demo
Watch the video demonstration of the Shape Classifier in action: [YouTube Video](https://youtu.be/l1Ufw39TTwM)

## Description
The Shape Classifier is a neural network model designed to distinguish between three basic geometric shapes: **circle**, **triangle**, and **square**. It takes grayscale images of dimensions **28 x 28 pixels** as input and predicts the shape present in the image.

This project can be a useful starting point for beginners in machine learning and deep learning to understand how neural networks can be used for image classification tasks.

## Features
- **Image Preprocessing:** Accepts grayscale images of size 28x28 pixels.
- **Shape Recognition:** Classifies images into one of three categories: circle, triangle, or square.
- **Lightweight and Efficient:** The model is designed to be simple and suitable for educational purposes.

## Prerequisites
To get started, ensure you have the following libraries installed:

- `tensorflow`
- `numpy`

You can install them using:
```bash
pip install tensorflow numpy
```

## How to Run the Program
1. Clone the repository:
   ```bash
   git clone https://github.com/abdullohtariq/shape_classifier.git
   cd shape_classifier
   ```

2. Ensure the required libraries are installed (see prerequisites).

3. Prepare your dataset:
   - Provide 28x28 pixel grayscale images of shapes (circle, triangle, square) as input.
   - Ensure the data is formatted correctly (e.g., as NumPy arrays).

4. Run the program:
   ```bash
   python shape_classifier.py
   ```

5. Follow the instructions in the terminal to classify shapes.

## How It Works
- **Dataset:** The model expects input images with shapes such as circles, triangles, and squares.
- **Neural Network Architecture:** A simple feedforward neural network built with TensorFlow.
- **Training:** The model is trained on a dataset of labeled shapes to identify patterns and features unique to each shape.
- **Prediction:** The trained model can then predict the shape in new unseen images.

## Folder Structure
```
shape_classifier/
├── shape_classifier.py   # Main program file
├── README.md             # Documentation
├── data/                 # Directory for input shape images
├── model/                # Trained model files (if any)
└── requirements.txt      # List of required Python libraries
```

## Example Usage
Here is an example of how the program works:
1. Input an image of a circle (28x28 pixels).
2. The program processes the image and runs it through the trained neural network.
3. The output will display:
   ```
   Predicted Shape: Circle
   ```

## Contribution
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

## License
This project is open source and available under the [MIT License](LICENSE).

## Author
Developed by [Abdulloh Tariq](https://github.com/abdullohtariq).
