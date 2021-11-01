# ml_6350
This is a machine learning library developed by Daniel Waldram for CS 5350/6350 in University of Utah.

To see instructions for assignment 2, look in the README in the Ensemble Learning folder.
To see instructions for assignment 3, look in the README in the Perceptron folder.


DecisionTree

	To train a decision tree use the id3 function in either id3alg.py or id3alg_p3.py in the 
	id3 inputs:
 	- prev_node_examples - the original examples list
 	- node_examples - the same input as prev_node_examples
	 - full_attributes - a dictionary of the attributes, which is recieved by passing node_examples to the get_attributes() function in id3alg.py
	 - remain_attributes - copy of full_attributes
 	- max_size - the maximum depth wanted for the decision tree
 	- gain_type - either "gini", "entropy", or "me" depending on desired gain method
 	 
  	To get the error from a set of training or test data use the test_data_error_calc function in either id3alg.py or id3alg_p3.py
  	decision_tree, examples inputs:
  	- decision_tree - decision tree output of id3
  	- examples - examples from which you want training error

EnsembleLearning

	ADABOOST (adaboost.py)
    adaboost - To run the adaboost function, you need to specify the number of models that will be used with adaboost as the first argument. The second argument is the list of training examples passed as a 2D list. This function returns a dictionary that holds "trees" and "votes" corresponding to those trees. 
    adaboost_get_predictions_error - To get the prediction error for adaboost, pass the dictionary recieved from adaboost as the first argument and examples for the second argument.
    classifier_weight - Does not need to be called by itself but gets called by the adaboost function. Updates the weight vector and determines the vote for a given tree.
    BAGGING (cs6350_hw2_p2b.py)
    process_trees - This function only takes in the number of trees that will be used and the list of full attributes. It also uses the training examples. This function returns the training and testing predictions for the dataset
    bag_get_predictions_error - This function takes in examples, predictions from the process_trees function, and the votes associated with those predictions. The function returns the average error

LinearRegression

	linear_regression (linear_regression.py) - this function takes in the example x_values, y_values, an initial weight vector and an r value. The function returns a weight vector and a vector that gives the cost for each epoch
    cost_function - This function takes in x_values, y_values, and a weight_vec. It returns the cost given those inputs
    stochastic_grad_descent (stochastic_grad_descent.py) - This function has the same inputs and outputs as the linear regression function but performs stochastic gradient descent

Perceptron (perceptron.py)

	prediction - takes in an example and a weight vector and returns a prediction as a 1 or -1
      voted_prediction - takes in an example and a dictionary of weight vectors and their associated votes and returns a prediction
      average_error - takes in examples and weight vector and returns average error. Calls prediction function
      average_error_voted - takes in examples and weight dictionary and returns average error. Calls voted_prediction function
      perceptron_alg - takes in examples, number of desired epochs, and learning rate and returns a weight vector
      voted_perceptron_alg - takes in same things as perceptron algorithm but returns a dictionary of weight vectors and their associated votes. Also returns the number of weight vectors
      average_perceptron_alg - takes in same things as perceptron algorithm. Returns a weight vector as well which represents the average weight vector. Also returns the final weight vector added to the average to be used for comparision with the perceptron_alg
      

