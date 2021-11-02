import math
import numpy as np
from perceptron import *
import csv

# Reading in the set of training examples
train = []
with open("bank/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# columns = train[0]
train_str_to_flt = example_numeric(train)
learn_rate = 0.01
w_final, num_w = voted_perceptron_alg(train_str_to_flt, 10, learn_rate)  # [0]*(len(train[0]) - 1)
print("weights          # correct predictions")
for row in range(num_w):
    print(w_final["weight_vectors"][row], "   ", w_final["votes"][row])
print("num weights: ", num_w)
# Reading in the set of test examples
test = []
with open("bank/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# columns = train[0]
test_str_to_flt = example_numeric(test)
test_error = average_error_voted(test_str_to_flt,  w_final)
print("Average Test Error: ", test_error)

weights_array = [z for z in w_final["weight_vectors"]]
for row in range(len(weights_array)):
    weights_array[row].append(w_final["votes"][row])
# print(weights_array[1:5])
# Write weights to the csv file
with open('weights.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write the header
    header = ["w_0", "w_1", "w_2", "w_3", "w_4", "correct count"]
    writer.writerow(header)
    # write a row to the csv file
    for row in range(len(weights_array)):
        writer.writerow(weights_array[row])
