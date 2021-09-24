from id3alg_p3 import *
import pandas as pd

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

gini_train_error = []
me_train_error = []
entropy_train_error = []
gini_test_error = []
me_test_error = []
entropy_test_error = []
tree_size = []

for i in range(1, 17):
    tree_size.append(i)
    # making decision tree of size i (1-17) for all gain methods
    dec_tree_entropy = id3(processed_train, processed_train, full_attributes, remain_attributes, i, "entropy")
    dec_tree_me = id3(processed_train, processed_train, full_attributes, remain_attributes, i, "me")
    dec_tree_gini = id3(processed_train, processed_train, full_attributes, remain_attributes, i, "gini")

    # calculating the training error
    entropy_train_error.append(test_data_error_calc(dec_tree_entropy, processed_train))
    me_train_error.append(test_data_error_calc(dec_tree_me, processed_train))
    gini_train_error.append(test_data_error_calc(dec_tree_gini, processed_train))
    # calculating the test error
    entropy_test_error.append(test_data_error_calc(dec_tree_entropy, processed_test))
    me_test_error.append(test_data_error_calc(dec_tree_me, processed_test))
    gini_test_error.append(test_data_error_calc(dec_tree_gini, processed_test))

print("-----PROBLEM 3A--------")
data = {'Entropy train': entropy_train_error, 'Entropy test': entropy_test_error, 'ME train': me_train_error, 'ME test': me_test_error, 'Gini train': gini_train_error, 'Gini test': gini_test_error}
df = pd.DataFrame(data, index=tree_size)
print(df)
