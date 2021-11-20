from scipy.optimize import minimize
import numpy as np
import math
from gaussian_kernel import *

'''
def gaussian_prediction(alpha, x, y_i, x_i, b):
    gaussian_sum = 0
    for i in range(len(x_i)):
        ximinx_sum = 0
        for j in range(len(x) - 1):
            ximinx_sum += (x_i[i][j] - x[j])**2
        gaussian_sum += alpha[i]*y_i[i]*math.exp(-ximinx_sum/5)
    gaussian_sum += gaussian_sum + b
    return math.copysign(1, gaussian_sum)


def gaussian_prediction_float(alpha, x, y_i, x_i):
    gaussian_sum = 0
    for i in range(len(x_i)):
        ximinx_sum = 0
        for j in range(len(x) - 1):
            ximinx_sum += (x_i[i][j] - x[j])**2
        gaussian_sum += alpha[i]*y_i[i]*math.exp(-ximinx_sum/5)
    gaussian_sum += gaussian_sum
    return gaussian_sum



# preprocessing converting strings to floats and adding on bias
def example_numeric(full_examples):
    examples = [z[:] for z in full_examples]
    # converting inputs to floats
    for row in range(len(examples)):
        for col in range(len(examples[0]) - 1):
            examples[row][col] = float(full_examples[row][col])
    # converting outputs to ints -1 or 1
    for row in range(len(examples)):
        if examples[row][len(examples[0])-1] == '0':
            examples[row][len(examples[0])-1] = -1
        else:
            examples[row][len(examples[0]) - 1] = 1
    # adding in a 1 for the bias term
    for row in range(len(examples)):
        examples[row].insert(-1, 1)
    return examples


# preprocessing converting strings to floats and adding on bias
def example_numeric_wo_bias(full_examples):
    examples = [z[:] for z in full_examples]
    # converting inputs to floats
    for row in range(len(examples)):
        for col in range(len(examples[0]) - 1):
            examples[row][col] = float(full_examples[row][col])
    # converting outputs to ints -1 or 1
    for row in range(len(examples)):
        if examples[row][len(examples[0])-1] == '0':
            examples[row][len(examples[0])-1] = -1
        else:
            examples[row][len(examples[0]) - 1] = 1
    return examples


# optimization function for the dual SVM
def gaussian_optimization_function(alpha_np, gaussian_mat, yyt):
    alpha_alphat = np.outer(alpha_np, alpha_np.T)
    y_alpha_gaus = yyt*alpha_alphat*gaussian_mat
    sum_total = 0.5*np.sum(y_alpha_gaus) - np.sum(alpha_np)
    return sum_total


# constraint function for the dual SVM
def constraint(alpha, y):
    return np.sum(y*alpha)


# average error calc for a set of examples
def average_error_gaussian(examples, alpha, b):
    error_sum = 0
    y_np = np.array(examples_np[:, -1])
    x_np = np.array(examples_np[:, :-1])
    for row in range(len(examples)):
        predict = gaussian_prediction(alpha, examples[row], y_np, x_np, b)
        #predict = prediction(examples[row], weight)
        # converting the prediction back to 0 to 1 so the error magnitude is consistent with expectations
        if predict == -1:
            predict = 0
        if examples[row][-1] == -1:
            answer = 0
        else:
            answer = 1
        error_sum += abs(predict - answer)
    return error_sum/len(examples)


# prediction
def prediction(example, weight):
    # sum with weights
    predict_sum = 0
    for col in range(len(weight)):
        predict_sum += example[col]*weight[col]
    return math.copysign(1, predict_sum)

'''
# Reading in the set of training examples
train = []
with open("bank-note/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# columns = train[0]
#print("Train: ", train[0])

sub_train = []
for i in range(150):
    sub_train.append(train[i])
    #print(sub_train[i])
# converting training examples to float
train_str_to_flt = example_numeric_wo_bias(train)
#print("Train float: ", train_str_to_flt[0])

# initializing alpha parameters to zero
initial_params = [0]*len(train_str_to_flt)
# convert to numpy
alpha_np = np.array(initial_params)
examples_np = np.array(train_str_to_flt)
y_np = np.array(examples_np[:, -1])
x_np = np.array(examples_np[:, :-1])

yyt = np.outer(y_np, y_np.T)

# assembling gaussian kernel
ones_x_np = np.ones(shape=x_np.shape)
x_2 = x_np**2
xixi = np.dot(x_2, ones_x_np.T)
xjxj = np.transpose(xixi)
xxt = np.dot(x_np, x_np.T)
gaussian_input = xixi - 2*xxt + xjxj

# setting value for C
C = 500/873
# bounds for the value of alpha
bound = (0, C)
bnds_list = []
for row in range(len(train_str_to_flt)):
    bnds_list.append(bound)
bnds_tuple = tuple(bnds_list)

# constraints for the values of alpha
constr = ({'type': 'eq', 'fun': constraint, 'args': (y_np,)})

# array to hold the alpha values for each pass
alphas = []
# array of gamma values to use
gammas = [0.1, 0.5, 1, 5, 100]

for gamma in gammas:
    print("gamma: ", gamma)
    lfunc = lambda e: math.exp(-e / gamma)
    myfunc_vec = np.vectorize(lfunc)
    gaussian_mat = myfunc_vec(gaussian_input)
    # minimization function call
    solution = minimize(gaussian_optimization_function, x0=alpha_np, args=(gaussian_mat, yyt), method='SLSQP', bounds=bnds_tuple, constraints=constr)
    alphas.append(solution.x)

threshold = 0.0001
# compute the number of overlapping support vectors
overlapping_support_vecs = [0]*(len(gammas) - 1)
for gamma in range(len(alphas) - 1):
    for row in range(len(alphas[0])):
        if alphas[gamma][row] >= threshold and alphas[gamma + 1][row] >= threshold:
            overlapping_support_vecs[gamma] += 1
    print("Overlapping Support Vectors between ", gammas[gamma], " and ", gammas[gamma + 1], ": ", overlapping_support_vecs[gamma])
