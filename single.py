import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generate_data():
    inches = np.linspace(-100, 100, num=500)
    centimeters = inches * 2.54
    return inches, centimeters

inches, centimeters = generate_data()

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(inches, centimeters, epochs=500, verbose=0)

test_values = np.array([5, 12, 50, 72.5, 99], dtype=float)
predicted_centimeters = model.predict(test_values)

for i, c in enumerate(test_values):
    print(f"Inches: {c} -> Predicted Centimeters: {predicted_centimeters[i][0]:.2f} (Actual: {c * 2.54})")