from random import random
import math
from neural_net import *


def main():
    # initially random weights are set for the nn
    neural_net = initialize_network(2, 3, 2)
    # weights are chosen here based on problem 3
    weights = [[[-1, -2, -3], [1, 2, 3]], [[-1, -2, -3], [1, 2, 3]], [[-1, 2, -1.5]]]
    neural_net = set_weights(neural_net, weights)
    print(neural_net)
    # forward pass given example [1, 1, 1]
    example = [1, 1, 1]
    neural_net = forward_pass(neural_net, example)
    # back propogation applied to example
    neural_net = back_propagation(neural_net, neural_net[len(neural_net) - 1][0]["prediction"], 1, example)
    print(neural_net)
    for layer in range(len(neural_net)):
        print("Layer", layer, ":")
        for node in range(len(neural_net[layer])):
            print(" node", node, " gradients: ", neural_net[layer][node]["grads"])


if __name__ == '__main__':
    main()
