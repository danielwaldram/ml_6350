from scipy.optimize import minimize
import numpy as np
import math
from gaussian_kernel import *

# Reading in the set of training examples
train = []
with open("bank-note/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))

sub_train = []
for i in range(150):
    sub_train.append(train[i])

# converting training examples to float
train_str_to_flt = example_numeric_wo_bias(train)

# Reading in the set of test examples
test = []
with open("bank-note/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))

test_str_to_flt = example_numeric_wo_bias(test)

# initializing alpha parameters to zero
initial_params = [0]*len(train_str_to_flt)
# convert to numpy
alpha_np = np.array(initial_params)
examples_np = np.array(train_str_to_flt)
y_np = np.array(examples_np[:, -1])
x_np = np.array(examples_np[:, :-1])
print(y_np.shape)
yyt = np.outer(y_np, y_np.T)

# assembling gaussian kernel
ones_x_np = np.ones(shape=x_np.shape)
x_2 = x_np**2
xixi = np.dot(x_2, ones_x_np.T)
xjxj = np.transpose(xixi)
xxt = np.dot(x_np, x_np.T)
gaussian_input = xixi - 2*xxt + xjxj

# setting value for C
C = [100/873, 500/873, 700/873]

# constraints for the values of alpha
constr = ({'type': 'eq', 'fun': constraint, 'args': (y_np,)})

# array of gamma values to use
gammas = [0.1, 0.5, 1, 5, 100]
for C_val in C:
    print("C value: ", C_val)
    # bounds for the value of alpha
    bound = (0, C_val)
    bnds_list = []
    for row in range(len(train_str_to_flt)):
        bnds_list.append(bound)
    bnds_tuple = tuple(bnds_list)
    for gamma in gammas:
        print(" Gamma value: ", gamma)
        # creating function for given gamma value
        lfunc = lambda e: math.exp(-e / gamma)
        myfunc_vec = np.vectorize(lfunc)
        # creating the gaussian matrix
        gaussian_mat = myfunc_vec(gaussian_input)
        # minimization function call
        solution = minimize(gaussian_optimization_function, x0=alpha_np, args=(gaussian_mat, yyt), method='SLSQP', bounds=bnds_tuple, constraints=constr)

        # computing the bias term
        threshold = 0.0001
        num_support_vecs = 0
        b_sum = 0
        for row in range(len(solution.x)):
            if solution.x[row] >= threshold:
                num_support_vecs += 1
                b_sum += train_str_to_flt[row][-1] - gaussian_prediction_float(solution.x, train_str_to_flt[row], y_np, x_np)

        bias = b_sum/num_support_vecs
        print("     bias: ", bias)
        print("     num vecs:", num_support_vecs)
        test_error = average_error_gaussian(test_str_to_flt, solution.x, bias, y_np, x_np)
        print('     test error: ', test_error)
        train_error = average_error_gaussian(train_str_to_flt, solution.x, bias, y_np, x_np)
        print('     train error: ', train_error)

