import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from svm import *


# Reading in the set of training examples
train = []
with open("bank-note/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# columns = train[0]
#print("Train: ", train[0])
train_str_to_flt = example_numeric(train)
#print("Train float: ", train_str_to_flt[0])

# Reading in the set of test examples
test = []
with open("bank-note/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# columns = train[0]
#print("Test: ", test[0])
test_str_to_flt = example_numeric(test)
#print("Test float: ", test_str_to_flt[0])

num_epochs = 100
#rate_0 = [0.01]
rate_0 = [0.00001, .00001, .00001]
C_val = [100/873, 500/873, 700/873]
a = [0.1, 0.1, 0.1]
#a = [100]

for val in range(len(C_val)):
    final_weight, cost_vec = primal_svm(train_str_to_flt, num_epochs, rate_0[val], "schedule1", C_val[val], a[val])
    magnitude = 0
    for weight in final_weight:
        magnitude += pow(weight, 2)
    magnitude = pow(magnitude, 0.5)
    #normalized_weight = 1/magnitude*final_weight
    print("Final weight: ", final_weight)
        #print("Normalized weight: ", normalized_weight)

    test_error = average_error(test_str_to_flt, final_weight)
    print("Average Test Error (C=", f"{C_val[val]:.3f}", "): ", test_error)
    train_error = average_error(train_str_to_flt, final_weight)
    print("Average Train Error (C=", f"{C_val[val]:.3f}", "): ", train_error)
    # plot the training results
    plt.plot(cost_vec)
plt.title("SVM: Stochastic Sub-Gradient Descent")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.legend(['C=100/873', 'C=500/873', 'C=700/873'])
plt.savefig('cost_combined.png')
plt.show()


# Repeat with the second schedule
rate_0 = [0.0001, 0.0001, 0.0001]

for val in range(len(C_val)):
    final_weight, cost_vec = primal_svm(train_str_to_flt, num_epochs, rate_0[val], "schedule2", C_val[val], a[val])
    magnitude = 0
    for weight in final_weight:
        magnitude += pow(weight, 2)
    magnitude = pow(magnitude, 0.5)
    #normalized_weight = 1/magnitude*final_weight
    print("Final weight: ", final_weight)
        #print("Normalized weight: ", normalized_weight)

    test_error = average_error(test_str_to_flt, final_weight)
    print("Average Test Error (C=", f"{C_val[val]:.3f}", "): ", test_error)
    train_error = average_error(train_str_to_flt, final_weight)
    print("Average Train Error (C=", f"{C_val[val]:.3f}", "): ", train_error)
    # plot the training results
    plt.plot(cost_vec)
plt.title("SVM: Stochastic Sub-Gradient Descent")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.legend(['C=100/873', 'C=500/873', 'C=700/873'])
plt.savefig('cost_combined_sch2.png')
plt.show()
