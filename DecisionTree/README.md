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
