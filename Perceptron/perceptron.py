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


# prediction
def prediction(example, weight):
    # sum with weights
    predict_sum = 0
    for col in range(len(weight)):
        predict_sum += example[col]*weight[col]
    return int(math.copysign(1, predict_sum))


# prediction with voting
def voted_prediction(example, weights):
    # sum with weights
    predict_sum = 0
    for vector in range(len(weights["weight_vectors"])):
        weight_predict = prediction(example, weights["weight_vectors"][vector])
        predict_sum += weight_predict*weights["votes"][vector]
    return int(math.copysign(1, predict_sum))


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


# average error calc for a set of examples using voted perceptron method
def average_error_voted(examples, weight):
    error_sum = 0
    for row_num in range(len(examples)):
        predict = voted_prediction(examples[row_num], weight)
        # converting the prediction back to 0 to 1 so the error magnitude is consistent with expectations
        if predict == -1:
            predict = 0
        if examples[row_num][-1] == -1:
            answer = 0
        else:
            answer = 1
        error_sum += abs(predict - answer)
    return error_sum/len(examples)


def perceptron_alg(examples, epochs, rate):
    # initialize weight vector
    weights = [0]*(len(examples[0]) - 1)
    for epoch in range(epochs):
        # randomize the examples order
        np.random.seed()
        rand_examples = []
        rand_nums = np.random.choice(len(examples), len(examples), replace=False)
        for index in range(len(rand_nums)):
            rand_examples.append(examples[rand_nums[index]])
        # Go through each example and update the weight vector
        for row in range(len(rand_examples)):
            # make a prediction
            predict = prediction(rand_examples[row], weights)
            if predict != rand_examples[row][-1]:
                for weight in range(len(weights)):
                    weights[weight] += rate*rand_examples[row][-1]*rand_examples[row][weight]
    return weights


# perceptron algorithm with voting
def voted_perceptron_alg(examples, epochs, rate):
    # initialize weight vector
    weights = {"weight_vectors": [[0]*(len(examples[0]) - 1)], "votes": [1]}
    # number of weight vectors
    m = 0
    # running for set number of epochs
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
            predict = prediction(rand_examples[row_num], weights["weight_vectors"][m])
            # print("prediction: ", predict)
            if predict != rand_examples[row_num][-1]:
                # create the new weight vector
                new_weights = []
                for weight in range(len(weights["weight_vectors"][m])):
                    new_weights.append(weights["weight_vectors"][m][weight] + rate*rand_examples[row_num][-1]*rand_examples[row_num][weight])
                # increase the number of votes
                m += 1
                weights["weight_vectors"].append(new_weights)
                weights["votes"].append(1)
            else:
                weights["votes"][m] += 1
    return weights, m


# perceptron algorithm with averaging
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
