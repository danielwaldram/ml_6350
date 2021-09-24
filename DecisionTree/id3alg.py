import math
import sys


class DecisionNode:
    def __init__(self, attribute, inherits):
        self.attribute = attribute
        self.inherits = inherits

    def __str__(self):
        return str(self.attribute)

    def __repr__(self):
        return str(self.attribute)


# takes in examples and returns the most common label among them
def most_common_label(examples):
    dict_of_labels = labels_count(examples)
    highest_count = 0
    highest_label = None
    for label in dict_of_labels:
        if dict_of_labels[label] >= highest_count:
            highest_count = dict_of_labels[label]
            highest_label = label
    return highest_label


def same_label_check(examples):
    first_label = examples[0][-1]
    for row in examples:
        if row[-1] != first_label:
            return False
    return True


def delete_examples(examples, att_val, attribute):
    new_examples = []
    for row in examples:
        if row[attribute] == att_val:
            new_examples.append(row)
    return new_examples


def id3(prev_node_examples, node_examples, full_attributes, remain_attributes, max_size, gain_type):
    #print(full_attributes)
    #print("remain attributes:")
    #print(remain_attributes)
    #print("node_examples:")
    #print(node_examples)
    #x = raw_input()
    # id3 will return this node represented as a dictionary
    node = {}
    # If there are no examples left. The examples from the previous node are used to determine the leaf's label
    if len(node_examples) == 0:
        # node is labeled a leaf
        node['subnodes'] = 'leaf'
        # leaf label is made most common from the previous nodes examples
        node['label'] = most_common_label(prev_node_examples)
        return node

    # Check if all the examples have the same label
    if same_label_check(node_examples):
        node['subnodes'] = 'leaf'
        # they're all the same, so I will make the first examples label the label for the leaf node
        node['label'] = node_examples[0][-1]
        return node

    # Check if there are any more attributes on which to split
    if len(remain_attributes.keys()) == 0:
        node['subnodes'] = 'leaf'
        # they're all the same, so I will make the first examples label the label for the leaf node
        node['label'] = most_common_label(node_examples)
        return node

    # Limit the size of the tree to the max size
    if max_size is not None:
        if (len(full_attributes) - len(remain_attributes)) == max_size:
            node['subnodes'] = 'leaf'
            # they're all the same, so I will make the first examples label the label for the leaf node
            node['label'] = most_common_label(node_examples)
            return node

    # Need to find the attribute that will best split the data out of all the remaining attributes
    if gain_type == 'entropy':
        best_attribute, best_gain = highest_info_gain_entropy(node_examples, remain_attributes)
    elif gain_type == 'me':
        best_attribute, best_gain = highest_info_gain_me(node_examples, remain_attributes)
    elif gain_type == 'gini':
        best_attribute, best_gain = highest_info_gain_gini(node_examples, remain_attributes)
    else:
        sys.exit("You entered an incorrect string for gain_type.")

    # Now know the attribute for the next split, so this can be saved along with the gain
    node['attribute'] = best_attribute
    node['subnodes'] = {}

    # need to remove the attribute from the remaining attributes
    new_remain_attributes = remain_attributes.copy()
    del new_remain_attributes[best_attribute]

    for attribute_val in full_attributes[best_attribute]:
        # need to delete any examples that don't fit this attribute value
        new_node_examples = delete_examples(node_examples, attribute_val, best_attribute)
        new_prev_node_examples = [x[:] for x in node_examples]
        node['subnodes'][attribute_val] = id3(new_prev_node_examples, new_node_examples, full_attributes, new_remain_attributes, max_size, gain_type)

    return node


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


def labels_count(examples):
    # Creating a dictionary to hold each label and associated count
    dict_of_labels = {}
    for i in range(len(examples)):
        # If the dictionary item already exists, then it's total is added to
        if examples[i][-1] in dict_of_labels:
            dict_of_labels[examples[i][-1]] = dict_of_labels.get(examples[i][-1]) + 1
        # If the dictionary item doesn't already exist, then it's total starts at 1
        else:
            dict_of_labels[examples[i][-1]] = 1
    return dict_of_labels


# Function that calculates the entropy of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def entropy_calc(examples):
    # get dictionary of labels
    dict_of_labels = labels_count(examples)

    # Calculating the entropy of the set by iterating through the dictionary adding up entropy contributions
    entropy = 0
    for key in dict_of_labels.keys():
        entropy = entropy - (float(dict_of_labels[key])/len(examples))*math.log(float(dict_of_labels[key])/len(examples), 2)
    return entropy


# Function that calculates the majority error of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def majority_error_calc(examples):
    # find the most common label
    label_most = most_common_label(examples)
    if label_most is None:
        return 0
    # get counts for every label in dictionary
    label_dict = labels_count(examples)
    # calculate the majority error
    maj_err = (len(examples) - label_dict[label_most])/float(len(examples))
    # return
    return maj_err


# Function that calculates the gini index of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def gini_index_calc(examples):
    # get dictionary of labels
    dict_of_labels = labels_count(examples)

    # Calculating the gini index of the set by iterating through the dictionary adding up probability contributions
    gi_sum = 0
    for key in dict_of_labels.keys():
        gi_sum = gi_sum + pow(float(dict_of_labels[key]) / len(examples), 2)
    gi = 1 - gi_sum
    return gi


# Function returns the highest information gain using ME method. Using all the given examples and all the given
#   attributes to determine which attribute splits the best
def highest_info_gain_gini(examples, attributes):
    # ME of the set of examples
    set_gi = gini_index_calc(examples)
    # dictionary holding all the information gain values
    information_gains = {}
    # highest information gain of the given attributes
    highest_info_gain = 0
    # iterating through each attribute provided
    for key in attributes.keys():
        # info gain for specific attribute value
        info_gain_sum = 0
        # split the example by attribute value
        sub_examples = attribute_split(examples, key, attributes)
        # iterating through every row (corresponding to an attribute value) of sub_examples to get contribution to gain
        for row in sub_examples:
            info_gain_sum = info_gain_sum + float(len(sub_examples[row])) / len(examples) * gini_index_calc(
                sub_examples[row])
        # Calculating the total information gain
        information_gains[key] = set_gi - info_gain_sum
        if information_gains[key] >= highest_info_gain:
            highest_info_gain_key = key
            highest_info_gain = information_gains[key]
    return highest_info_gain_key, highest_info_gain


# Function returns the highest information gain using ME method. Using all the given examples and all the given
#   attributes to determine which attribute splits the best
def highest_info_gain_me(examples, attributes):
    # ME of the set of examples
    set_me = majority_error_calc(examples)
    # dictionary holding all the information gain values
    information_gains = {}
    # highest information gain of the given attributes
    highest_info_gain = 0
    # iterating through each attribute provided
    for key in attributes.keys():
        # info gain for specific attribute value
        info_gain_sum = 0
        # split the example by attribute value
        sub_examples = attribute_split(examples, key, attributes)
        # iterating through every row (corresponding to an attribute value) of sub_examples to get contribution to gain
        for row in sub_examples:
            info_gain_sum = info_gain_sum + float(len(sub_examples[row])) / len(examples) * majority_error_calc(
                sub_examples[row])
        # Calculating the total information gain
        information_gains[key] = set_me - info_gain_sum
        if information_gains[key] >= highest_info_gain:
            highest_info_gain_key = key
            highest_info_gain = information_gains[key]
    return highest_info_gain_key, highest_info_gain


# Function returns the highest information gain using entropy method. Using all the given examples and all the given
#   attributes to determine which attribute splits the best
def highest_info_gain_entropy(examples, attributes):
    # entropy of the set of examples
    set_entropy = entropy_calc(examples)
    # dictionary holding all the information gain values
    information_gains = {}
    # highest information gain of the given attributes
    highest_info_gain = 0
    # iterating through each attribute provided
    for key in attributes.keys():
        # info gain for specific attribute value
        info_gain_sum = 0
        # split the example by attribute value
        sub_examples = attribute_split(examples, key, attributes)
        # iterating through every row (corresponding to an attribute value) of sub_examples to get contribution to gain
        for row in sub_examples:
            info_gain_sum = info_gain_sum + float(len(sub_examples[row]))/len(examples)*entropy_calc(sub_examples[row])
        # Calculating the total information gain
        information_gains[key] = set_entropy - info_gain_sum

        if information_gains[key] >= highest_info_gain:
            highest_info_gain_key = key
            highest_info_gain = information_gains[key]
    return highest_info_gain_key, highest_info_gain


# Function to split the examples up by attribute value. Returns a dictionary with an entry for every attribute value and
#   a list for all the rows under it.
def attribute_split(examples, key, attributes):
    # dictionary to hold all the example labels split up by the value of the current attribute of interest
    sub_examples = {}
    # temp_examples is the examples list but it gets smaller as its looped through to improve efficiency
    temp_examples = examples[:][:]
    # iterating through the list of possible values under the given attribute
    for k in range(len(attributes[key])):
        # a new row in sub_examples is saved for the current attribute value
        sub_examples[attributes[key][k]] = []
        # index for the temporary examples array
        j = 0
        for i in range(len(temp_examples)):
            # if the example has the same attribute value as the current one of interest, then the examples label
            #   is added to its row of the sub_examples array. If not then the temporary examples index "j" is
            #   increased.
            if temp_examples[j][key] == attributes[key][k]:
                sub_examples[attributes[key][k]].append(temp_examples[j][-1])
                temp_examples.remove(temp_examples[j])
            else:
                j = j + 1
    return sub_examples


# This function will take in a decision_tree and an example without a label.
#   the tree then makes a prediction on what the label of the example is
def decision_tree_predictor(decision_tree, example):
    if 'attribute' in decision_tree:
        # determine what attribute is split by this tree
        attribute = decision_tree["attribute"]
        # the attribute number should correspond to the index number in the example
        value = example[attribute]
        # finding the branch specific to the value for the attribute
        value_branch = decision_tree['subnodes'][value]
        label = decision_tree_predictor(value_branch, example)
        return label
    return decision_tree['label']


def decision_tree_batch_predictor(decision_tree, examples):
    prediction = []
    for row in examples:
        prediction.append(decision_tree_predictor(decision_tree, row))
    return prediction


def test_data_error_calc(decision_tree, examples):
    # first I need to get rid of the labels
    test_data_no_label = [z[:-1] for z in examples]
    test_prediction = decision_tree_batch_predictor(decision_tree, test_data_no_label)
    true_value = [z[-1] for z in examples]
    error_total = 0
    for i in range(len(examples)):
        if test_prediction[i] != true_value[i]:
            error_total += 1
    average_error = float(error_total)/len(examples)
    return average_error
