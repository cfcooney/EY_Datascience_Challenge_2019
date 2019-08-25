"""
Name: Ciaran Cooney
Date: 05/04/2019
Description: Useful function for implementing a voting system among multiple classifiers.
"""

def most_frequent(List):
	"""
	Returns most frequent element in list
	"""
	return max(set(List), key = List.count) 

def voting(x,y,z):
	"""
	Returns the most common prediction given multiple predictors.

	"""
	pred = []
	for i in zip(x,y,z):
	    pred.append(most_frequent(i))
	return pred

    