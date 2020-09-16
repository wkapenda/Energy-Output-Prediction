import numpy as np
import pandas as pd

# Read data from the excel file and store it into a dataset
dataset = pd.read_csv('Folds5x2_pp.csv')

x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation= None))

ann.compile(optimizer='adam', loss='mean_squared_error')

ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print(y)