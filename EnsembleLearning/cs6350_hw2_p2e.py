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
    # Creating a new set of examples that is random: uniform with replacement
    np.random.seed()
    rand_examples = []
    rand_nums = np.random.choice(len(processed_train), len(processed_train))
    for index in range(len(rand_nums)):
        rand_examples.append(processed_train[rand_nums[index]])
    # print('Model: ', iteration)

    # Randomly sample a very small subset of the attributes
    # random choose attributes
    np.random.seed()
    rand_nums = np.random.choice(len(list(full_attributes.keys())), downsampled_att_num, replace=False)
    downsampled_attributes = {}
    for i in rand_nums:
        downsampled_attributes[i] = full_attributes[i]

    downsampled_remain_attributes = downsampled_attributes.copy()

    new_tree = id3(rand_examples, rand_examples, downsampled_attributes, downsampled_remain_attributes, None, "entropy")

    testing_predictions = []
    # get predictions for the testing set
    examples_no_label = [z[:-1] for z in processed_test]
    for row in examples_no_label:
        testing_predictions.append(decision_tree_predictor(new_tree, row))
    # Return the training and testing predictions for the given model
    return testing_predictions


# Wrapper so that I can call process_trees from a concurrent.futures.ProcessPoolExecutor instance
def process_trees_wrapper(p):
    return process_trees(*p)


def main(downsampled_attribute_num):

    # getting all the attributes that are in examples
    full_attributes = get_attributes(processed_train_5000)
    # number of iterations
    num_models = 500

    # Getting predictions from each individual model for testing and training and saving to these lists
    #   doing this rather than saving all the models will make getting average error faster and take less memory
    testing_predictions = []
    num_models_list = list(range(1, num_models + 1))
    args = ((model_num, full_attributes, downsampled_attribute_num) for model_num in num_models_list)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_trees_wrapper, args)

    for result in results:
        testing_predictions.append(result)
    return testing_predictions


if __name__ == "__main__":
    # Reading in the set of training examples
    train = []
    with open("bank/train.csv", 'r') as f:
        for line in f:
            train.append(line.strip().split(','))

    processed_train_5000, numerical_medians = numerical_train_data_preprocessing(train)

    # Reading in the set of test examples
    test = []
    with open("bank/test.csv", 'r') as f:
        for line in f:
            test.append(line.strip().split(','))

    processed_test = numerical_test_data_preprocessing(test, numerical_medians)

    for multiplier in range(1, 4):
        attribute_num = multiplier*2
        bag_predictions = []
        for step in range(100):
            print("Bag Algorithm # ", step)
            # Creating a new set of examples that is random: uniform without replacement
            np.random.seed()
            processed_train = []
            random_nums = np.random.choice(len(processed_train_5000), 1000, replace=False)

            for i in range(len(random_nums)):
                processed_train.append(processed_train_5000[random_nums[i]])

            bag_predictions.append(main(attribute_num))

        # Predictions are yes and no so I will convert to 1 = yes and 0 = no so I can do statistics more easily
        for bag_alg in range(len(bag_predictions)):
            for predic in range(len(bag_predictions[bag_alg])):
                for label in range(len(bag_predictions[bag_alg][predic])):
                    if bag_predictions[bag_alg][predic][label] == 'yes':
                        bag_predictions[bag_alg][predic][label] = 1
                    elif bag_predictions[bag_alg][predic][label] == 'no':
                        bag_predictions[bag_alg][predic][label] = 0
                    else:
                        print("ERROR")

        test_cases_label = [z[-1] for z in processed_test]
        # convert test cases to numerical as well
        for test_case in range(len(test_cases_label)):
            if test_cases_label[test_case] == 'yes':
                test_cases_label[test_case] = 1
            elif test_cases_label[test_case] == 'no':
                test_cases_label[test_case] = 0
            else:
                print("ERROR")

        var_list = []
        bias_list = []
        # Compute the predictions of the 100 single trees
        print("Single Tree Predictions")
        for answer in range(len(test_cases_label)):
            # get the average prediction
            sum_predictions = 0
            var_sum = 0
            # Finding the average prediction for a given example
            for bag in range(len(bag_predictions)):
                # Look at the answer for first tree in the bag
                sum_predictions += bag_predictions[bag][0][answer]
            average_prediction = sum_predictions/len(bag_predictions)
            # Finding the variance for the given sample
            for bag in range(len(bag_predictions)):
                # Look at the answer for first tree in the bag
                var_sum += pow(bag_predictions[bag][0][answer] - average_prediction, 2)
            # variance and bias for a particular sample
            var_list.append(var_sum/(len(bag_predictions) - 1))
            bias_list.append(pow(average_prediction - test_cases_label[answer], 2))

        var_sum = 0
        bias_sum = 0
        for item in range(len(var_list)):
            var_sum += var_list[item]
            bias_sum += bias_list[item]

        average_var = var_sum/len(var_list)
        average_bias = bias_sum/len(bias_list)

        print("Avg Var Single Random Tree " + str(attribute_num) + " Attributes: ", average_var)
        print("Avg Bias Single Random Tree " + str(attribute_num) + " Attributes: ", average_bias)
        print("Avg Gen Squared Error " + str(attribute_num) + " Attributes: ", average_var + average_bias)

        # Compute the predictions of the 100-500 tree bags
        print("500 Tree Predictions")
        # Get the prediction for each
        var_list = []
        bias_list = []
        for answer in range(len(test_cases_label)):
            # get the average prediction
            sum_predictions = 0
            var_sum = 0
            alg_pred = []
            # Finding the average prediction for a given example
            for bag in range(len(bag_predictions)):
                # Need to get the estimate for the given bagging algorithm
                sub_tree_sum = 0
                for sub_tree in range(len(bag_predictions[bag])):
                    sub_tree_sum += bag_predictions[bag][sub_tree][answer]
                # Get the average prediction from the specific bagging algorithm
                av_sub_tree_pred = sub_tree_sum/len(bag_predictions[bag])
                # Determine if yes or no got the most votes and assign that value as the prediction of the bag
                if av_sub_tree_pred >= 0.5:
                    alg_pred.append(1)
                else:
                    alg_pred.append(0)
                sum_predictions += alg_pred[bag]
            # get the average prediction for all the bag algorithms
            average_prediction = sum_predictions / len(bag_predictions)
            # Finding the variance for the given sample
            for bag in range(len(bag_predictions)):
                # Look at the answer for first tree in the bag
                var_sum += pow(alg_pred[bag] - average_prediction, 2)
            # variance and bias for a particular sample
            var_list.append(var_sum / (len(bag_predictions) - 1))
            bias_list.append(pow(average_prediction - test_cases_label[answer], 2))

        var_sum = 0
        bias_sum = 0
        for item in range(len(var_list)):
            var_sum += var_list[item]
            bias_sum += bias_list[item]

        average_var = var_sum / len(var_list)
        average_bias = bias_sum / len(bias_list)

        print("Avg Var Random Trees " + str(attribute_num) + " Attributes: ", average_var)
        print("Avg Bias Random Trees " + str(attribute_num) + " Attributes: ", average_bias)
        print("Avg Gen Squared Error " + str(attribute_num) + " Attributes: ", average_var + average_bias)
