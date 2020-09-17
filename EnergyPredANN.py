import numpy as np
import pandas as pd

# Read data from the excel file and store it into a dataset
dataset = pd.read_csv('Folds5x2_pp.csv')

# Create the features (x) and target (y) data sets
x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# import sklearn machine learning library
from sklearn.model_selection import train_test_split

# Split the features and target datasets into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Import the tensorflow machine learning library to create the ANN model
import tensorflow as tf

# Initialize the ANN model
ann = tf.keras.models.Sequential()

# Add dense layers to create an ANN model with an architecture of 4-6-6-1
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation= None))

# Compile the ANN model using the adam optimizer and the mean squared error loss function
ann.compile(optimizer='adam', loss='mean_squared_error')

# Fit the training data into the ANN model, using epochs = 100
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Use the test data to make the prediction
y_pred = ann.predict(X_test)

# Set the precision to two decimal places
np.set_printoptions(precision=2)

new_y_test = y_test.reshape(len(y_test), 1)
#print(new_y_test)
#print(y_pred)

# Calculate the accuracy of the ANN model
model_acc = 100-((abs(y_pred - new_y_test)/new_y_test)*100)

print(model_acc)

print(np.average(model_acc))

# Accuracy of 99.144 %
