import math


def id3(examples):
    attributes = get_attributes(examples)
    print(entropy_calc(examples))
    information_gain(examples, attributes)


# Function returns a dictionary where the keys are the attributes as numbers 0 to attribute count.
#   The values of the dictionary are the possible attribute values.
def get_attributes(examples):
    dict_of_attributes = {}
    for j in range(len(examples[0]) - 1):
        dict_of_attributes[j] = []
        for i in range(len(examples)):
            if examples[i][j] not in dict_of_attributes[j]:
                dict_of_attributes[j].append(examples[i][j])
    return dict_of_attributes


# Function that calculates the entropy of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def entropy_calc(examples):
    # Creating a dictionary to hold each label and associated count
    dict_of_labels = {}
    for i in range(len(examples)):
        # If the dictionary item already exists, then it's total is added to
        if examples[i][-1] in dict_of_labels:
            dict_of_labels[examples[i][-1]] = dict_of_labels.get(examples[i][-1]) + 1
        # If the dictionary item doesn't already exist, then it's total starts at 1
        else:
            dict_of_labels[examples[i][-1]] = 1

    # Calculating the entropy of the set by iterating through the dictionary adding up entropy contributions
    entropy = 0
    for key in dict_of_labels.keys():
        entropy = entropy - (float(dict_of_labels[key])/len(examples))*math.log(float(dict_of_labels[key])/len(examples), 2)
    return entropy


def information_gain(examples, attributes):
    info_gain_sum = 0
    information_gains = []
    # iterating through each attribute provided
    for key in attributes.keys():
        # array to hold all the example labels split up by the value of the current attribute of interest
        sub_examples = []
        # temp_examples is the examples array but it gets smaller as its looped through to improve efficiency
        temp_examples = examples[:][:]
        # iterating through the list of possible values under the given attribute
        for k in range(len(attributes[key])):
            # a new row in sub_examples is saved for the current attribute value
            sub_examples.append([])
            # index for the temporary examples array
            j = 0
            for i in range(len(temp_examples)):
                # if the example has the same attribute value as the current one of interest, then the examples label
                #   is added to its row of the sub_examples array. If not then the temporary examples index "j" is
                #   increased.
                if temp_examples[j][key] == attributes[key][k]:
                    sub_examples[k].append(temp_examples[j][-1])
                    temp_examples.remove(temp_examples[j])
                else:
                    j = j+1
    print(sub_examples)

terms = []
with open("car/train.csv", 'r') as f:
    for line in f:
        terms.append(line.strip().split(','))

id3(terms)
