import math
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


# Cost function takes in the x_values, y_values, and weight_vec and determines a cost
def cost_function(x_values, y_values, weight_vec):
    cost = 0
    for row in range(len(x_values)):
        prediction = 0
        for col in range(len(weight_vec)):
            prediction += weight_vec[col]*x_values[row][col]
        try:
            cost += pow(y_values[row] - prediction, 2)
        except OverflowError:
            print("Overflow")
            return 0.5*cost
    return 0.5*cost

# Reading in the set of training examples
train = []
with open("concrete/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))

# Converting data to float
for row in range(len(train)):
    for col in range(len(train[row])):
        train[row][col] = float(train[row][col])

# Split the training data into x and y
x_values = [z[:-1] for z in train]
y_values = [z[-1] for z in train]

# add bias onto the x_values as the bias term
for row in range(len(x_values)):
    x_values[row].append(1)

x_val = np.array(x_values)
y_val = np.transpose(np.array(y_values))
xmult = np.matmul(np.transpose(x_val), x_val)
w_vec = np.matmul(np.matmul(np.linalg.inv(xmult),np.transpose(x_val)), y_val)
weight_vec = np.ndarray.tolist(w_vec)
print(w_vec)
print('cost: ', cost_function(x_values, y_values, weight_vec))

