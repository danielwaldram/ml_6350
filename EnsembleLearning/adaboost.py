import math
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from id3alg_weighted import *


# this function will calculate a classifiers weight based on the adaboost algorithm
def classifier_weight(decision_tree, weight_vector, examples):
    # First find the total error
    # getting rid of labels
    test_data_no_label = [z[:-1] for z in examples]
    test_prediction = decision_tree_batch_predictor(decision_tree, test_data_no_label)
    true_value = [z[-1] for z in examples]
    error_total = 0
    # add up the error
    for i in range(len(examples)):
        if test_prediction[i] != true_value[i]:
            error_total += weight_vector[i]
    # get the tree's vote
    # this is a trivial case, which I accounted for to avoid errors
    if error_total == 0:
        vote = 1
    else:
        vote = 1/2*math.log((1-error_total)/error_total)
    # create the new weight vector
    new_weight_vec = []
    for i in range(len(examples)):
        if test_prediction[i] != true_value[i]:
            new_weight_vec.append(weight_vector[i]*math.exp(vote))
        else:
            new_weight_vec.append(weight_vector[i]*math.exp(-vote))

    weight_mag = sum(new_weight_vec)
    new_weight_vec = [x/weight_mag for x in new_weight_vec]
    return new_weight_vec, vote


def adaboost(num_models, examples):
    # Creating an initial weight vector
    weight_vector = [1 / len(examples)] * len(examples)
    # Creating a dictionary to hold a list of votes and associated decision trees
    ada_boost_models = {"trees": [], "votes": []}
    # getting all the attributes that are in examples
    full_attributes = get_attributes(examples)
    # remaining attributes variable is passed to id3 because it will change on recursion, it will remain the same
    #   for this initial call
    remain_attributes = full_attributes.copy()

    for i in range(num_models):
        print("Model ", i)
        # create a decision tree stump
        dec_tree_entropy = id3(examples, examples, full_attributes, remain_attributes, weight_vector, weight_vector, 1, "entropy")
        # determine the weight for the stump
        weight_vector, vote = classifier_weight(dec_tree_entropy, weight_vector, examples)
        ada_boost_models["trees"].append(dec_tree_entropy)
        ada_boost_models["votes"].append(vote)
    return ada_boost_models


def adaboost_get_predictions_error(ada_models, examples):
    predictions = []
    # getting rid of labels
    examples_no_label = [z[:-1] for z in examples]
    # going through every example
    for row in range(len(examples_no_label)):
        votes = {}
        # Go through all the models to get their predictions
        for model in range(len(ada_models["votes"])):
            guess = decision_tree_predictor(ada_models["trees"][model], examples_no_label[row])
            # add to the vote total for the given prediction
            if guess in votes:
                votes[guess] += ada_models["votes"][model]
            else:
                votes[guess] = ada_models["votes"][model]
        highest_vote = 0
        # determine which label had the highest vote total
        for label in votes:
            if votes[label] >= highest_vote:
                prediction = label
                highest_vote = votes[label]
        predictions.append(prediction)
    # get the average error
    true_value = [z[-1] for z in examples]
    error_total = 0
    for i in range(len(examples)):
        if predictions[i] != true_value[i]:
            error_total += 1
    average_error = float(error_total)/len(examples)
    return average_error
