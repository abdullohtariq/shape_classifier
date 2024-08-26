from data_preparation import load_data
from model import create_model

# Load data
X_train, X_test, y_train, y_test = load_data('data')

# Create model
model = create_model()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('shape_classifier_model.h5')
