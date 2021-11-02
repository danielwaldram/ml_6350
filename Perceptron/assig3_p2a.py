import math
import numpy as np
from perceptron import *

# Reading in the set of training examples
train = []
with open("bank/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# columns = train[0]
train_str_to_flt = example_numeric(train)
learn_rate = 0.01
w_final = perceptron_alg(train_str_to_flt, 10, learn_rate)
print("w final: ", w_final)

# Reading in the set of test examples
test = []
with open("bank/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# columns = train[0]
test_str_to_flt = example_numeric(test)
test_error = average_error(test_str_to_flt, w_final)
print("Average Test Error: ", test_error)
