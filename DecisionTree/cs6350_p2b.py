from id3alg import *

# Reading in the set of training examples
train = []
with open("car/train.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))

# getting all the attributes that are in examples
full_attributes = get_attributes(train)
# remaining attributes is passed to id3 because it will change on recursion, it will remain the same
#   for this initial call
remain_attributes = full_attributes.copy()

# Reading in the set of test examples
test = []
with open("car/test.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))


gini_train_error = []
me_train_error = []
entropy_train_error = []
gini_test_error = []
me_test_error = []
entropy_test_error = []

for i in range(1, 7):
    # making decision tree of size i (1-6) for all gain methods
    dec_tree_entropy = id3(train, train, full_attributes, remain_attributes, i, "entropy")
    dec_tree_me = id3(train, train, full_attributes, remain_attributes, i, "me")
    dec_tree_gini = id3(train, train, full_attributes, remain_attributes, i, "gini")

    # calculating the training error
    entropy_train_error.append(test_data_error_calc(dec_tree_entropy, train))
    me_train_error.append(test_data_error_calc(dec_tree_me, train))
    gini_train_error.append(test_data_error_calc(dec_tree_gini, train))
    # calculating the test error
    entropy_test_error.append(test_data_error_calc(dec_tree_entropy, test))
    me_test_error.append(test_data_error_calc(dec_tree_me, test))
    gini_test_error.append(test_data_error_calc(dec_tree_gini, test))

print("Entropy Training Errors:")
print(entropy_train_error)
print("Majority Error Training Errors:")
print(me_train_error)
print("Gini Index Training Errors:")
print(gini_train_error)

print("Entropy Test Errors:")
print(entropy_test_error)
print("Majority Error Test Errors:")
print(me_test_error)
print("Gini Index Test Errors:")
print(gini_test_error)

