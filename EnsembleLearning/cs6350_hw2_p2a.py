import math
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from adaboost import *


def main():
    # Reading in the set of training examples
    train = []
    with open("bank/train.csv", 'r') as f:
        for line in f:
            train.append(line.strip().split(','))
    processed_train, numerical_medians = numerical_train_data_preprocessing(train)
    # getting all the attributes that are in examples
    full_attributes = get_attributes(processed_train)
    # remaining attributes is passed to id3 because it will change on recursion, it will remain the same
    #   for this initial call
    remain_attributes = full_attributes.copy()

    # Reading in the set of test examples
    test = []
    with open("bank/test.csv", 'r') as f:
        for line in f:
            test.append(line.strip().split(','))

    processed_test = numerical_test_data_preprocessing(test, numerical_medians)

    
    training_error_boost = []
    training_error_indv = []
    testing_error_boost = []
    num_of_models = 500
    models = adaboost(num_of_models, processed_train)
    # print(models)
    # Get training error for each model
    for i in range(1, num_of_models + 1):
        model_subgroup = {"trees": [], "votes": []}
        model_subgroup["trees"].extend(models["trees"][0:i])
        model_subgroup["votes"].extend(models["votes"][0:i])
        #print('subgroup votes: ', model_subgroup["votes"])
        training_error_indv.append(test_data_error_calc(models['trees'][i - 1], processed_train))
        training_error_boost.append(adaboost_get_predictions_error(model_subgroup, processed_train))
        testing_error_boost.append(adaboost_get_predictions_error(model_subgroup, processed_test))
    print("Training Error:", training_error_boost)
    print("Training Error Individual :", training_error_indv)
    plt.plot(training_error_boost)
    plt.plot(testing_error_boost)
    plt.xlabel("iteration")
    plt.ylabel("average error")
    plt.title("Boosting Error")
    plt.savefig('boosting_error_tst.png')
    plt.show()
    plt.plot(training_error_indv)
    plt.title("Individual Stump Error")
    plt.xlabel("iteration")
    plt.ylabel("average error")
    plt.savefig('stump_error_tst.png')
    plt.show()


if __name__ == "__main__":
    main()
