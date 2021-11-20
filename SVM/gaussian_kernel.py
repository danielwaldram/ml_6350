from scipy.optimize import minimize
import numpy as np
import math


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
    return 0.5*np.sum(y_alpha_gaus) - np.sum(alpha_np)


# constraint function for the dual SVM
def constraint(alpha, y):
    return np.sum(y*alpha)


# average error calc for a set of examples
def average_error_gaussian(examples, alpha, b, y_np, x_np):
    error_sum = 0
    np_examples = np.array(examples)
    for row in range(len(examples)):
        predict = gaussian_prediction(alpha, np_examples[row], y_np, x_np, b)
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
