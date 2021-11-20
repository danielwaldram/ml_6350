from scipy.optimize import minimize
import numpy as np
import math
from dual_svm import *


# Reading in the set of training examples
train = []
with open("bank-note/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# columns = train[0]
print("Train: ", train[0])

# converting training examples to float
train_str_to_flt = example_numeric_wo_bias(train)
print("Train float: ", train_str_to_flt[0])

# Reading in the set of test examples
test = []
with open("bank-note/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# columns = train[0]
#print("Test: ", test[0])
test_str_to_flt = example_numeric(test)

# initializing alpha parameters to zero
initial_params = [0]*len(train_str_to_flt)
# convert to numpy
alpha_np = np.array(initial_params)
examples_np = np.array(train_str_to_flt)
y_np = np.array(examples_np[:, -1])
x_np = np.array(examples_np[:, :-1])
# complete the x and y matrices that go into the optimization function
xxt = np.dot(x_np, x_np.T)
yyt = np.outer(y_np, y_np.T)

# setting value for C
C = [100/873, 500/873, 700/873]

# constraints for the values of alpha
constr = ({'type': 'eq', 'fun': constraint, 'args': (y_np,)})

# Running the SVM for each value of C
for value in C:
    # bounds for the value of alpha
    bound = (0, value)
    bnds_list = []
    for row in range(len(train_str_to_flt)):
        bnds_list.append(bound)
    bnds_tuple = tuple(bnds_list)

    print("C value: ", value)
    # minimization function call
    solution = minimize(optimization_function, x0=alpha_np, args=(yyt, xxt), method='SLSQP', bounds=bnds_tuple, constraints=constr)

    # converting alpha values to a single weight vector, without bias
    weight_vec = np.zeros(len(train_str_to_flt[0]) - 1)
    for weight in range(len(weight_vec)):
        for row in range(len(train_str_to_flt)):
            weight_vec[weight] += solution.x[row]*train_str_to_flt[row][-1]*train_str_to_flt[row][weight]

    # print the weight vector
    print("weight vector: ", weight_vec)

    # computing the bias term
    threshold = 0.0001
    num_support_vecs = 0
    b_sum = 0
    for row in range(len(solution.x)):
        if solution.x[row] >= threshold:
            num_support_vecs += 1
            weight_sum = 0
            for weight in range(len(weight_vec)):
                weight_sum += weight_vec[weight]*train_str_to_flt[row][weight]
            b_sum += train_str_to_flt[row][-1] - weight_sum

    bias = b_sum/num_support_vecs
    print("bias: ", bias)
    print("num vecs:", num_support_vecs)
    # add the bias onto the weight vector
    weight_vec_full = np.append(weight_vec, bias)

    test_error = average_error(test_str_to_flt, weight_vec_full)
    print('test error: ', test_error)
