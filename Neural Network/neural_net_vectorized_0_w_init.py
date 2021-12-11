import random
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


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


def create_weight_mat(input_len, layer_width):
    matrix = np.array([0 for _ in range(input_len * layer_width)])
    # put in matrix form
    matrix = matrix.reshape((layer_width, input_len))
    # add the last row
    last_row = np.zeros((1, input_len))
    last_row[0, -1] = 1
    return np.append(matrix, last_row, axis=0)


# network is initialized with a certain number of hidden layers and nodes within those layers
# layer width does not include the bias term. Input length does include the bias term
def initialize_network(layer_width, input_len, hidden_layers):
    # first layer of nodes has weights equal to the number of inputs
    initial_layer = create_weight_mat(input_len, layer_width)
    neural_net = [{"weights": initial_layer, "predictions": np.zeros((1, input_len)), "grads": np.zeros((layer_width, input_len))}]
    # all other layers have weights equal to the layer width
    for i in range(1, hidden_layers):
        neural_net.append({"weights": create_weight_mat(layer_width + 1, layer_width), "predictions": np.append(np.zeros((1, layer_width)), 1), "grads": np.zeros((layer_width, layer_width + 1))})
    # output layer is always a single node
    neural_net.append({"weights": np.array([random.gauss(0, 1) for _ in range(layer_width + 1)]), "predictions": 0, "grads": np.array((layer_width + 1)*[0.0])})
    return neural_net


# define new weights for the neural net with a list of weight lists that has the same structure as the neural net
def set_weights(neural_net, weights):
    for layer in range(len(neural_net)):
        neural_net[layer]["weights"] = weights[layer]
    return neural_net


# forward_pass through the network gives a prediction for an example
def forward_pass(neural_net, example):
    #print("example: ", example)
    # initialize the initial prediction based on the specific example. This gets passed to the first layer
    neural_net[0]["predictions"] = np.array(example).T
    # go through every layer in the neural net except the last
    for layer in range(len(neural_net)-1):
        # finding the prediction for the next layer
        neural_net[layer + 1]["s_vals"] = np.matmul(neural_net[layer]["weights"], neural_net[layer]["predictions"])
        #print(neural_net[layer + 1]["s_vals"])
        neural_net[layer + 1]["predictions"] = np.append(np.array(list(map(activation_function, neural_net[layer + 1]["s_vals"][0:-1]))), 1)
    # final layer is predicted by a linear combination of inputs without an activation
    neural_net[len(neural_net) - 1]["output"] = np.dot(neural_net[len(neural_net) - 1]["weights"], neural_net[len(neural_net) - 1]["predictions"])
    return neural_net


# back propogation algorithm needs to take in the neural_net, the prediction from a forward pass of the nn for an example, the correct value for that example, and the example
def back_propagation(neural_net, prediction, true_value, example):
    # pass back is initialized
    pass_back = [prediction - true_value]
    # first layer
    #print(pass_back)
    #print(neural_net[len(neural_net) - 1]["predictions"])
    neural_net[len(neural_net) - 1]['grads'] = pass_back*neural_net[len(neural_net) - 1]["predictions"]
    #print(neural_net[len(neural_net) - 1]['grads'])
    for layer in range(len(neural_net) - 2, -1, -1):
        input_vals = neural_net[layer]["predictions"]
        #print("weights", neural_net[layer + 1]["weights"])
        # cut the bias out from weights ahead
        if layer == len(neural_net) - 2:
            weights_ahead = neural_net[layer + 1]["weights"][0:-1]
        else:
            weights_ahead = np.delete(neural_net[layer + 1]["weights"], -1, axis=0)
            weights_ahead = np.delete(weights_ahead, -1, axis=1)
        #print(weights_ahead)
        s_prime_vals = np.array(list(map(activation_prime, neural_net[layer + 1]["s_vals"][0:-1])))
        #print("s_prime_vals: ", s_prime_vals)
        #print(np.ones((1, len(neural_net[layer + 1]["predictions"] - 1))))
        #print("weights ahead: ", weights_ahead)
        if layer == len(neural_net) - 2:
            s_prime_mat = np.diag(s_prime_vals)
        else:
            s_prime_mat = np.diag(s_prime_vals)
        der_addition = np.matmul(s_prime_mat, weights_ahead)
        #print("pass back: ", pass_back)
        if layer == len(neural_net) - 2:
            pass_back = pass_back*der_addition
        else:
            pass_back = np.matmul(pass_back, der_addition)
        #print(der_addition)
        #print(layer)
        #print(pass_back)
        #print("input", input_vals)
        gradient_vec = np.outer(input_vals, pass_back)
        #print(gradient_vec.T)
        neural_net[layer]['grads'] = gradient_vec.T
    return neural_net


# sigmoid activation function used
def activation_function(activation_input):
    return 1.0 / (1.0 + np.exp(-activation_input))


def activation_prime(activation_prime_input):
    return activation_function(activation_prime_input)*(1 - activation_function(activation_prime_input))


def cost_calc(neural_net, examples):
    loss_sum = 0
    # run through every example
    for example in examples:
        nn_example = forward_pass(neural_net, example[:-1])
        loss_sum += 0.5*(nn_example[-1]['output'] - example[-1])**2
    return loss_sum


def stochastic_gradient_descent_nn(examples, epochs, neural_net, rate0, a):
    cost_vector = []
    last_row_input_layer = np.zeros((1, len(examples[0]) - 1))
    last_row_input_layer[0, -1] = 1
    last_row_hidden_layer = np.zeros((1, len(neural_net[1]["predictions"])))
    last_row_hidden_layer[0, -1] = 1
    #print("last row input: ", last_row_input_layer)

    for epoch in range(epochs):
        cost_vector.append(cost_calc(neural_net, examples))
        # update the rate
        rate = rate0 / (1 + (rate0 / a) * epoch)
        # randomize the examples order
        np.random.seed()
        # array of random numbers corresponding to rows in examples
        rand_nums = np.random.choice(len(examples), len(examples), replace=False)
        # Go through each example and update the weight vector
        for row in range(len(examples)):
            rand_row = rand_nums[row]
            #print("rand_row: ", examples[rand_row])
            # complete a forward pass to update predictions
            neural_net = forward_pass(neural_net, examples[rand_row][:-1])
            # complete back propagation to update the gradients
            #print("prediction: ", neural_net[-1][0]['prediction'])
            #print(len(neural_net))
            neural_net = back_propagation(neural_net=neural_net, prediction=neural_net[-1]['output'], true_value=examples[rand_row][-1], example=examples[rand_row][:-1])
            for layer in range(len(neural_net)):
                if layer == len(neural_net) - 1:
                    neural_net[layer]["weights"] = np.subtract(neural_net[layer]["weights"], rate * neural_net[layer]["grads"])
                elif layer == 0:
                    #print("last row input: ", last_row_input_layer)
                    #print("grads: ", neural_net[layer]["grads"])
                    #print("weights: ", neural_net[layer]["weights"])
                    #print("neural net", neural_net)
                    neural_net[layer]["weights"] = np.subtract(neural_net[layer]["weights"], rate * np.append(neural_net[layer]["grads"], last_row_input_layer, axis=0))
                else:
                    #print("weights", neural_net[layer]["weights"])
                    #print("grads",  neural_net[layer]["grads"])
                    #print(last_row_hidden_layer)
                    neural_net[layer]["weights"] = np.subtract(neural_net[layer]["weights"], rate * np.append(neural_net[layer]["grads"], last_row_hidden_layer, axis=0))
    return neural_net, cost_vector


# average error calc for a set of examples
def average_error(examples, neural_net):
    error_sum = 0
    # run through every example
    for example in examples:
        nn_example = forward_pass(neural_net, example[:-1])
        error_sum += abs(nn_example[-1]['output'] - example[-1])
    return error_sum/len(examples)


# initially random weights are set for the nn
neural_net = initialize_network(layer_width=100, input_len=5, hidden_layers=2)
# Reading in the set of training examples
train = []
with open("bank-note/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))

train_str_to_flt = example_numeric(train)

# Reading in the set of test examples
test = []
with open("bank-note/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))

test_str_to_flt = example_numeric(test)

a_vals = [0.2, 1, 0.2, 0.2, 0.3]
lr_vals = [0.5, 0.2, 0.15, 0.15, 0.05]
width = [5, 10, 25, 50, 100]

for network in range(len(width)):
    print("Width: ", width[network])
    # initially random weights are set for the nn
    neural_net = initialize_network(layer_width=width[network], input_len=5, hidden_layers=2)
    neural_net, cost_calc_vec = stochastic_gradient_descent_nn(train_str_to_flt, 100, neural_net, lr_vals[network], a_vals[network])
    # calculating average test and training error
    training_error = average_error(train_str_to_flt, neural_net)
    testing_error = average_error(test_str_to_flt, neural_net)
    print("training error " + str(width[network]) + " Nodes: " + str(training_error))
    print("testing error " + str(width[network]) + " Nodes: " + str(testing_error))

