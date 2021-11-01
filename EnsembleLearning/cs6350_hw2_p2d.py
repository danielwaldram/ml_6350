import math
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import concurrent.futures
from id3alg import *


# Function takes votes from previous iterations of the bagging algorithm and adds on votes from the current tree
#   the error is calculated and passed back out along with the updated votes
def bag_get_predictions_error(votes, predictions_new_tree, examples):
    predictions_final = []
    # getting rid of labels
    examples_no_label = [z[:-1] for z in examples]
    # going through every example
    for row in range(len(examples_no_label)):
        # Go through the new model and get its predictions
        guess = predictions_new_tree[row]
        if guess in votes[row]:
            votes[row][guess] += 1
        else:
            votes[row][guess] = 1
        highest_vote = 0
        # determine which label had the highest vote total
        for label in votes[row]:
            if votes[row][label] >= highest_vote:
                prediction = label
                highest_vote = votes[row][label]
        predictions_final.append(prediction)
    # get the average error
    true_value = [z[-1] for z in examples]
    error_total = 0
    for i in range(len(examples)):
        if predictions_final[i] != true_value[i]:
            error_total += 1
    average_error = float(error_total)/len(examples)
    return average_error, votes


# function to randomly sort the examples and process them
def process_trees(iteration, full_attributes, downsampled_att_num):
    remain_attributes = full_attributes.copy()
    # Creating a new set of examples that is random: uniform with replacement
    np.random.seed()
    rand_examples = []
    rand_nums = np.random.choice(len(processed_train), len(processed_train))
    for index in range(len(rand_nums)):
        rand_examples.append(processed_train[rand_nums[index]])
    print('Model: ', iteration)

    # Randomly sample a very small subset of the attributes
    # random choose attributes
    np.random.seed()
    rand_nums = np.random.choice(len(list(full_attributes.keys())), downsampled_att_num, replace=False)
    downsampled_attributes = {}
    for i in rand_nums:
        downsampled_attributes[i] = full_attributes[i]

    downsampled_remain_attributes = downsampled_attributes.copy()

    new_tree = id3(rand_examples, rand_examples, downsampled_attributes, downsampled_remain_attributes, None, "entropy")

    # get the predictions for the training set
    examples_no_label = [z[:-1] for z in processed_train]
    training_predictions = []
    testing_predictions = []
    for row in examples_no_label:
        training_predictions.append(decision_tree_predictor(new_tree, row))
    # get predictions for the testing set
    examples_no_label = [z[:-1] for z in processed_test]
    for row in examples_no_label:
        testing_predictions.append(decision_tree_predictor(new_tree, row))
    # Return the training and testing predictions for the given model
    return training_predictions, testing_predictions


# Wrapper so that I can call process_trees from a concurrent.futures.ProcessPoolExecutor instance
def process_trees_wrapper(p):
    return process_trees(*p)


def main(downsampled_attribute_num):
    # getting all the attributes that are in examples
    full_attributes = get_attributes(processed_train)

    # number of iterations
    num_models = 500

    # Getting predictions from each individual model for testing and training and saving to these lists
    #   doing this rather than saving all the models will make getting average error faster and take less memory
    testing_predictions = []
    training_predictions = []
    num_models_list = list(range(1, num_models + 1))
    args = ((model_num, full_attributes, downsampled_attribute_num) for model_num in num_models_list)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_trees_wrapper, args)

    for result in results:
        training_predictions.append(result[0])
        testing_predictions.append(result[1])

    # Creating a dictionary to hold the votes as the models get added up
    votes_train_dict = {}
    for i in range(len(processed_train)):
        votes_train_dict[i] = {}
    votes_test_dict = {}
    for i in range(len(processed_train)):
        votes_test_dict[i] = {}

    training_error_bag = []
    testing_error_bag = []

    # This loop gets the average error for both training and testing for each bagging algorithm.
    #   Saving votes allows me to pass information from previous iterations forward, so I don't need to recalc votes
    for i in range(num_models):
        average_error_train, votes = bag_get_predictions_error(votes_train_dict, training_predictions[i], processed_train)
        average_error_test, votes = bag_get_predictions_error(votes_test_dict, testing_predictions[i], processed_test)
        training_error_bag.append(average_error_train)
        testing_error_bag.append(average_error_test)

    print("Training Error:", training_error_bag)
    print("Testing Error:", testing_error_bag)

    plt.plot(training_error_bag, label='training')
    plt.plot(testing_error_bag, label='testing')
    plt.xlabel("iteration")
    plt.ylabel("average error")
    plt.title("Random Forest Error: " + str(downsampled_attribute_num) + " Attributes")
    plt.legend()
    plt.savefig('random_tree_' + str(downsampled_attribute_num) + 'att.png')
    plt.show()


if __name__ == "__main__":
    # Reading in the set of training examples
    train = []
    with open("bank/train.csv", 'r') as f:
        for line in f:
            train.append(line.strip().split(','))
    processed_train, numerical_medians = numerical_train_data_preprocessing(train)
    # Reading in the set of test examples
    test = []
    with open("bank/test.csv", 'r') as f:
        for line in f:
            test.append(line.strip().split(','))

    processed_test = numerical_test_data_preprocessing(test, numerical_medians)
    attribute_number = 2
    main(attribute_number)
    attribute_number = 4
    main(attribute_number)
    attribute_number = 6
    main(attribute_number)
