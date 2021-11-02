import math
import numpy as np
from perceptron import *


# perceptron algorithm with voting
def average_perceptron_alg(examples, epochs, rate):
    # initialize weight vector
    weights = [0] * (len(examples[0]) - 1)
    a = [z for z in weights]
    for epoch in range(epochs):
        # randomize the examples order
        np.random.seed()
        rand_examples = []
        rand_nums = np.random.choice(len(examples), len(examples), replace=False)
        for index in range(len(rand_nums)):
            rand_examples.append(examples[rand_nums[index]])
        # Go through each example and update the weight vector
        for row_num in range(len(rand_examples)):
            # make a prediction
            predict = prediction(rand_examples[row_num], weights)
            if predict != rand_examples[row_num][-1]:
                for weight in range(len(weights)):
                    weights[weight] += rate * rand_examples[row_num][-1] * rand_examples[row_num][weight]
                    a[weight] += weights[weight]
            for weight in range(len(weights)):
                a[weight] += weights[weight]
    return a, weights


# Reading in the set of training examples
train = []
with open("bank/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# columns = train[0]
train_str_to_flt = example_numeric(train)
learn_rate = 0.01
w_final, last_weights = average_perceptron_alg(train_str_to_flt, 10, learn_rate)  # [0]*(len(train[0]) - 1)
print("Final w: ", w_final)
# Reading in the set of test examples
test = []
with open("bank/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# columns = train[0]
test_str_to_flt = example_numeric(test)
test_error = average_error(test_str_to_flt,  w_final)
print("Average Test Error: ", test_error)
