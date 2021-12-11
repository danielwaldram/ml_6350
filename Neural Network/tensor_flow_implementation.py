from neural_net import *
import csv
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Reading in the set of training examples
train = []
with open("bank-note/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
columns = train[0][:-1]#['a', 'b', 'c', 'd']

train_str_to_flt = example_numeric(train)
train_array = np.array(train_str_to_flt, dtype=object)
mean0 = np.mean(train_array[:, 0])
train_array[:, 0] -= mean0
std0 = np.std(train_array[:, 0])
train_array[:, 0] /= std0
mean1 = np.mean(train_array[:, 1])
train_array[:, 1] -= mean1
std1 = np.std(train_array[:, 1])
train_array[:, 1] /= std1
mean2 = np.mean(train_array[:, 2])
train_array[:, 2] -= mean2
std2 = np.std(train_array[:, 2])
train_array[:, 2] /= std2
mean3 = np.mean(train_array[:, 3])
train_array[:, 3] -= mean3
std3 = np.std(train_array[:, 3])
train_array[:, 3] /= std3

# Reading in the set of training examples
test = []
with open("bank-note/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))

test_str_to_flt = example_numeric(test)
test_array = np.array(test_str_to_flt, dtype=object)
test_array[:, 0] -= mean0
test_array[:, 0] /= std0
test_array[:, 1] -= mean1
test_array[:, 1] /= std1
test_array[:, 2] -= mean2
test_array[:, 2] /= std2
test_array[:, 3] -= mean3
test_array[:, 3] /= std3

# Use test data for validation and split validation and training data
XVALIDATION = test_array[:, :-1]
YVALIDATION = test_array[:, -1]
XTRAIN = train_array[:, :-1]
YTRAIN = train_array[:, -1]

# getting y-values
YTRAIN = YTRAIN.astype(int)
YVALIDATION = YVALIDATION.astype(int)
train_array_sub = XTRAIN
# train_array_sub = np.concatenate((np.transpose([XTRAIN[:, 0]]), np.transpose([XTRAIN[:, 1]]), np.transpose([XTRAIN[:, 2]]), np.transpose([XTRAIN[:, 3]])), axis=1)
#print("train array sub: ", train_array_sub)

train_array_y = np.transpose([YTRAIN])
validation_array_sub = XVALIDATION
#validation_array_sub = np.concatenate((np.transpose([XVALIDATION[:, 0]]), np.transpose([XVALIDATION[:, 1]]), np.transpose([XVALIDATION[:, 2]]), np.transpose([XVALIDATION[:, 3]])), axis=1)
validation_array_y = np.transpose([YVALIDATION])

# data frame is made up of only the 1st row which is categorical in order to build as simple a test as possible
train_df = pd.DataFrame(data=train_array_sub, columns=columns)
validation_df = pd.DataFrame(data=validation_array_sub, columns=columns)

numeric_input = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input2 = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input3 = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input4 = layers.Input(shape=(1,), dtype=tf.float32)


concat = layers.concatenate([numeric_input, numeric_input2, numeric_input3, numeric_input4])

#model = models.Model(inputs=[numeric_input, categorical_input], outputs=[concat])
preprocessing_model = models.Model(inputs=[numeric_input, numeric_input2, numeric_input3, numeric_input4], outputs=concat)
#preprocessing_model = models.Model(inputs=categorical_input, outputs=encoded)
#predicted = model.predict(train_df[columns[0]], train_df[columns[1]])
#train_df[columns[0]] = pd.to_numeric(train_df[columns[0]])

# initialization method for the weights
initializer = tf.keras.initializers.HeNormal()  # He initialization for relu
x_initializer = tf.keras.initializers.GlorotNormal()  # xavier initialization for tanh
# Model for making predictions
train_hot_layer_transformed = preprocessing_model.predict([train_df[columns[0]].astype(float), train_df[columns[1]].astype(float), train_df[columns[2]].astype(float), train_df[columns[3]].astype(float)])
validation_hot_layer_transformed = preprocessing_model.predict([validation_df[columns[0]].astype(float), validation_df[columns[1]].astype(float), validation_df[columns[2]].astype(float), validation_df[columns[3]].astype(float)])

# depth and width values to be ran through
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ['relu', 'tanh']
initializers = [initializer, x_initializer]
callback_a = ModelCheckpoint(filepath='my_best_mode.hdf5', monitor='val_loss', save_best_only=True)

training_err_relu = []
testing_err_relu = []
training_err_tanh = []
testing_err_tanh = []

for depth in depths:
    for width in widths:
        for activation in range(len(activations)):
            # the model to be ran
            model = models.Sequential()
            #relu_layer = layers.Dense(width, activation='relu', kernel_initializer=initializer)
            model.add(layers.Dense(width, input_dim=len(train_hot_layer_transformed[0, :]), activation=activations[activation], kernel_initializer=initializers[activation]))
            for i in range(depth - 1):
                model.add(layers.Dense(width, activation=activations[activation], kernel_initializer=initializers[activation]))
            model.add(layers.Dense(1, activation='linear', kernel_initializer=initializers[activation]))
            model.summary()

            # Using mse to assess loss and Adam as the optimizer
            model.compile(loss='mse', optimizer="Adam", metrics=['accuracy'])

            history = model.fit(train_hot_layer_transformed, train_array_y, validation_data=(validation_hot_layer_transformed, validation_array_y), batch_size=10, epochs=100, callbacks=[callback_a])

            # get the predictions for the model for both training and testing
            predictions = model.predict(validation_hot_layer_transformed)
            training_predictions = model.predict(train_hot_layer_transformed)
            # get the testing and the training error
            testing_error = average_error(test_array, predictions)
            print('testing error: ', testing_error)
            training_error = average_error(train_array, training_predictions)
            print('training error: ', training_error)

            if activations[activation] == 'relu':
                training_err_relu.append(training_error)
                testing_err_relu.append(testing_error)
            if activations[activation] == 'tanh':
                training_err_tanh.append(training_error)
                testing_err_tanh.append(testing_error)

# printing out the errors
for depth in range(len(depths)):
    print("depth " + str(depths[depth]) + ":")
    for width in range(len(widths)):
        print("     width " + str(widths[width]) + ":")
        print('         RELU testing error: ', testing_err_relu[depth*width + depth])
        print('         RELU training error: ', training_err_relu[depth*width + depth])
        print('         tanh testing error: ', testing_err_tanh[depth * width + depth])
        print('         tanh training error: ', training_err_tanh[depth * width + depth])
