import math
import numpy as np


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


# cost calculation
def cost_calc(examples, weight_vector, C):
    empirical_cost = 0
    for row in range(len(examples)):
        predict = prediction_svm(examples[row], weight_vector)
        if predict*examples[row][-1] <= 1:
            empirical_cost += 1 - predict*examples[row][-1]
    empirical_cost = C*empirical_cost
    weight_cost = 0
    for weight in range(len(weight_vector) - 1):
        weight_cost += weight_vector[weight]
    cost = weight_cost + empirical_cost
    return cost


# prediction
def prediction_svm(example, weight):
    # sum with weights
    predict_sum = 0
    for col in range(len(weight)):
        predict_sum += example[col]*weight[col]
    return predict_sum


# prediction
def prediction(example, weight):
    # sum with weights
    predict_sum = 0
    for col in range(len(weight)):
        predict_sum += example[col]*weight[col]
    return math.copysign(1, predict_sum)


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


def primal_svm(examples, epochs, rate0, rate_mode, C, a):
    # initialize weight vector
    weights = [0]*(len(examples[0]) - 1)
    cost_vector = []
    for epoch in range(epochs):
        # update the rate
        if rate_mode == "schedule1":
            rate = rate0/(1 + (rate0/a)*epoch)
        if rate_mode == "schedule2":
            rate = rate0/(1 + epoch)
        # get the cost for the weight vector
        cost_vector.append(cost_calc(examples, weights, C))
        # randomize the examples order
        np.random.seed()
        rand_examples = []
        # array of random numbers corresponding to rows in examples
        rand_nums = np.random.choice(len(examples), len(examples), replace=False)

        for index in range(len(rand_nums)):
            # assemble randomly sorted examples
            rand_examples.append(examples[rand_nums[index]])
        # Go through each example and update the weight vector
        for row in range(len(rand_examples)):
            prediction = prediction_svm(rand_examples[row], weights)
            if prediction*rand_examples[row][-1] <= 1:
                for weight in range(len(weights) - 1):
                    weights[weight] = weights[weight] - rate*(weights[weight] - C*len(rand_examples)*rand_examples[row][-1]*rand_examples[row][weight])
                weights[-1] = rate*C*len(rand_examples)*rand_examples[row][-1]*rand_examples[row][-2]
            else:
                for weight in range(len(weights) - 1):
                    weights[weight] = (1 - rate)*weights[weight]
    return weights, cost_vector
