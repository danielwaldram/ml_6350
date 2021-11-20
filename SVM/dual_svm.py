from scipy.optimize import minimize
import numpy as np
import math


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
def optimization_function(alpha_np, yyt, xxt):
    alpha_alphat = np.outer(alpha_np, alpha_np.T)
    y_alpha_x = 0.5*yyt*alpha_alphat*xxt
    return np.sum(y_alpha_x) - np.sum(alpha_np)


# constraint function for the dual SVM
def constraint(alpha, y):
    return np.sum(y*alpha)


# average error calc for a set of examples
def average_error(examples, weight):
    error_sum = 0
    for row in range(len(examples)):
        predict = prediction(examples[row], weight)
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
