import random
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
        examples[row].insert(0, 1)
    return examples


# network is initialized with a certain number of hidden layers and nodes within those layers
# layer width does not include the bias term
def initialize_network(layer_width, input_len, hidden_layers):
    # first layer of nodes has weights equal to the number of inputs
    neural_net = [[{"weights": np.array([random.gauss(0, 1) for _ in range(input_len)]), "prediction": 0, "grads": np.array(input_len*[0.0])} for _ in range(layer_width)]]
    # all other layers have weights equal to the layer width
    for i in range(1, hidden_layers):
        neural_net.append([{"weights": np.array([random.gauss(0, 1) for _ in range(layer_width + 1)]), "prediction": 0, "grads": np.array((layer_width + 1)*[0.0])} for _ in range(layer_width)])
    # output layer is always a single node
    neural_net.append([{"weights": np.array([random.gauss(0, 1) for _ in range(layer_width + 1)]), "prediction": 0, "grads": np.array((layer_width + 1)*[0.0])}])
    return neural_net


# define new weights for the neural net with a list of weight lists that has the same structure as the neural net
def set_weights(neural_net, weights):
    for layer in range(len(neural_net)):
        for node in range(len(neural_net[layer])):
            neural_net[layer][node]["weights"] = weights[layer][node]
    return neural_net


# forward_pass through the network gives a prediction for an example
def forward_pass(neural_net, example):
    # initialize the prediction based on the specific example. This gets passed to the first layer
    prediction = example
    # go through every layer in the neural net except the last
    for layer in range(len(neural_net) - 1):
        # go through every node in the layer to get the new prediction list
        # new prediction starts with 1 for the bias term
        new_prediction = [1]
        for node in range(len(neural_net[layer])):
            #print("node: ", node)
            neural_net[layer][node]["prediction"] = activation_function(np.dot(neural_net[layer][node]["weights"], prediction))
            new_prediction.append(neural_net[layer][node]["prediction"])
        prediction = new_prediction


    # final layer is predicted by a linear combination of inputs without an activation
    neural_net[len(neural_net) - 1][0]["prediction"] = np.dot(neural_net[len(neural_net) - 1][0]["weights"], prediction)
    return neural_net


# back propogation algorithm needs to take in the neural_net, the prediction from a forward pass of the nn for an example, the correct value for that example, and the example
def back_propagation(neural_net, prediction, true_value, example):
    output_grad = prediction - true_value
    # get the gradient for the set of weights that go to the output node
    neural_net[len(neural_net) - 1][0]["grads"][0] = output_grad
    for node in range(len(neural_net[len(neural_net) - 2])):
        #print("prediction= ", neural_net[len(neural_net) - 2][node]["prediction"])
        #print("grads: ", neural_net[len(neural_net) - 1][0]["grads"][node + 1])
        neural_net[len(neural_net) - 1][0]["grads"][node + 1] = output_grad*neural_net[len(neural_net) - 2][node]["prediction"]

    pass_along = [1]
    # count down from the second to last layer
    for layer in range(len(neural_net) - 2, -1, -1):
        new_pass_along = []
        for node in range(len(neural_net[layer])):
            # number of terms in pass along determines how many additions are part of the current layer
            for term in range(len(pass_along)):
                new_pass_along.append(pass_along[term]*neural_net[layer + 1][term]["weights"][node + 1]*(neural_net[layer][node]["prediction"]*(1 - neural_net[layer][node]["prediction"])))
            for weight in range(len(neural_net[layer][node]["weights"])):
                gradient_sum = 0
                # number of terms in pass along determines how many additions are part of the current layer
                for term in range(len(pass_along)):
                    # multiply the pass along term by the weight 1 layer up associated with the current node and the current activation function derivative
                    if weight == 0:
                        gradient_sum += pass_along[term]*neural_net[layer + 1][term]["weights"][node + 1]*(neural_net[layer][node]["prediction"]*(1 - neural_net[layer][node]["prediction"]))
                    else:
                        if layer == 0:
                            gradient_sum += pass_along[term] * neural_net[layer + 1][term]["weights"][node + 1] * (neural_net[layer][node]["prediction"] * (1 - neural_net[layer][node]["prediction"])) * example[weight]
                        else:
                            gradient_sum += pass_along[term] * neural_net[layer + 1][term]["weights"][node + 1] * (neural_net[layer][node]["prediction"] * (1 - neural_net[layer][node]["prediction"]))*neural_net[layer - 1][weight - 1]["prediction"]
                neural_net[layer][node]["grads"][weight] = output_grad*gradient_sum
        # update the pass along term
        pass_along = new_pass_along
    return neural_net


# sigmoid activation function used
def activation_function(activation_input):
    return 1.0 / (1.0 + np.exp(-activation_input))


def cost_calc(neural_net, examples):
    loss_sum = 0
    # run through every example
    for example in examples:
        nn_example = forward_pass(neural_net, example[:-1])
        loss_sum += 0.5*(nn_example[-1][0]['prediction'] - example[-1])**2
    return loss_sum


def stochastic_gradient_descent_nn(examples, epochs, neural_net, rate0, a):
    cost_vector = []
    for epoch in range(epochs):
        print('epoch: ', epoch)
        cost_vector.append(cost_calc(neural_net, examples))
        print("cost: ", cost_vector[-1])
        # update the rate
        rate = rate0 / (1 + (rate0 / a) * epoch)
        print("rate: ", rate)
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
            neural_net = back_propagation(neural_net=neural_net, prediction=neural_net[-1][0]['prediction'], true_value=examples[rand_row][-1], example=examples[rand_row][:-1])
            for layer in range(len(neural_net)):
                for node in range(len(neural_net[layer])):
                    neural_net[layer][node]["weights"] = np.subtract(neural_net[layer][node]["weights"], rate*neural_net[layer][node]["grads"])
                    #for weight in range(len(neural_net[layer][node]["weights"])):
                    #    neural_net[layer][node]["weights"][weight] = neural_net[layer][node]["weights"][weight] - rate*neural_net[layer][node]["grads"][weight]
    return neural_net, cost_vector


# average error calc for a set of examples
def average_error(examples, neural_net):
    error_sum = 0
    # run through every example
    for example in examples:
        nn_example = forward_pass(neural_net, example[:-1])
        error_sum += abs(nn_example[-1][0]['prediction'] - example[-1])
    return error_sum/len(examples)
