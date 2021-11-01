import math
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


# Cost function takes in the x_values, y_values, and weight_vec and determines a cost
def cost_function(x_values, y_values, weight_vec):
    cost = 0
    for row in range(len(x_values)):
        prediction = 0
        for col in range(len(weight_vec)):
            prediction += weight_vec[col]*x_values[row][col]
        try:
            cost += pow(y_values[row] - prediction, 2)
        except OverflowError:
            print("Overflow")
            return 0.5*cost
    return 0.5*cost


def stochastic_grad_descent(x_values, y_values, weight_vec, r_value):
    # get the initial cost and add to the cost vector
    cost_vec = [cost_function(x_values, y_values, weight_vec)]
    #cost_vec = []
    # Loop through until the weight vector converges or max iterations is reached
    max_iter = 10000
    for iteration in range(max_iter):
        # randomize the examples
        rand_x_values = []
        rand_y_values = []
        np.random.seed()
        # Choose randomly without replacement
        rand_nums = np.random.choice(len(x_values), len(x_values), replace=False)
        for index in range(len(rand_nums)):
            rand_x_values.append(x_values[rand_nums[index]])
            rand_y_values.append(y_values[rand_nums[index]])

        # iterate through the examples
        for example in range(len(rand_x_values)):
            #input("Advance?")
            #print("Rand example: ", rand_x_values[example])
            # get the gradient of the cost function
            del_J = []
            for weight in range(len(weight_vec)):
                prediction = 0
                for col in range(len(rand_x_values[0])):
                    prediction += weight_vec[col] * rand_x_values[example][col]
                del_J.append(-(rand_y_values[example] - prediction)*rand_x_values[example][weight])

            # print("del J: ", del_J)
            # Update the weight vector
            w_new = []
            for weight in range(len(weight_vec)):
                w_new.append(weight_vec[weight] - r_value*del_J[weight])
            # Add the cost to the cost vector
            cost_vec.append(cost_function(x_values, y_values, w_new))
            # Check the change from weight vector to new weight vector
            w_sum_squared = 0
            for weight in range(len(weight_vec)):
                try:
                    w_sum_squared += pow(w_new[weight] - weight_vec[weight], 2)
                except OverflowError:
                    continue
            w_norm = pow(w_sum_squared, 0.5)
            # Update the weight vector based on the new weight vector
            weight_vec = [z for z in w_new]
            # Check if the weight vector has converged enough to exit
            if w_norm <= pow(10, -9):
                print("Iteration: ", iteration)
                print("w norm: ", w_norm)
                return weight_vec, cost_vec
    print("Iteration: ", iteration)
    print("w norm: ", w_norm)
    return weight_vec, cost_vec


def main():
    # Reading in the set of training examples
    train = []
    with open("concrete/train.csv", 'r') as f:
        for line in f:
            train.append(line.strip().split(','))

    # Converting data to float
    for row in range(len(train)):
        for col in range(len(train[row])):
            train[row][col] = float(train[row][col])

    # Split the training data into x and y
    x_values = [z[:-1] for z in train]
    y_values = [z[-1] for z in train]
    # Choose a rate
    r = 0.005
    # initializing the weight vector: one for each x value plus one for the bias
    weight_vec = [0]*(len(x_values[0]) + 1)

    # add bias onto the x_values as the bias term
    for row in range(len(x_values)):
        x_values[row].append(1)

    # Test linear regression
    final_weight, cost_vec = stochastic_grad_descent(x_values, y_values, weight_vec, r)

    # plot the training results
    plt.plot(cost_vec)
    plt.title("Stochastic Gradient Descent")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.savefig('stochastic_gradient_descent' + str(r) + '.png')
    plt.show()

    # get the cost of the test data
    # Reading in the set of test examples
    test = []
    with open("concrete/test.csv", 'r') as f:
        for line in f:
            test.append(line.strip().split(','))

    # Converting data to float
    for row in range(len(test)):
        for col in range(len(train[row])):
            test[row][col] = float(test[row][col])

    # Split the training data into x and y
    test_x_values = [z[:-1] for z in test]
    test_y_values = [z[-1] for z in test]

    # add bias onto the x_values as the bias term
    for row in range(len(test_x_values)):
        test_x_values[row].append(1)

    test_cost = cost_function(test_x_values, test_y_values, final_weight)
    print("Test cost: ", test_cost)
    print("Final weight vector: ", final_weight)


if __name__ == "__main__":
    main()
