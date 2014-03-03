import numpy as np 
import os

'''
	Given a matrix, calculate the mean and standard deviations for columns
	Standardize the matrix elements by applying the z-score formula:
	(x - mean)/std_deviation
'''
def standardize(matrix):
	rows = len(matrix)
	cols = len(matrix[0])

	feature_averages = matrix.mean(axis=0)	# Unidimensional array containing means for each column
	feature_std_devs = matrix.std(axis=0)	# Unidimensional array containing std_devs for each column

	for i in xrange(0, rows):
		for j in xrange(0, cols):
			standardized_val = (matrix[i, j] - feature_averages[j])/(feature_std_devs[j]) # Compute z-score
			matrix[i, j] = standardized_val # Set element to its computed z-score

	return matrix


if __name__ == '__main__':
	data_file = os.getcwd() + "/spam.data"
	matrix = np.genfromtxt(data_file)[:3065, :57] # For this experiment, I'm using this as my training data
	
	standardized_matrix = standardize(matrix)

	# Check that mean is about 0
	print standardized_matrix.mean(axis=0)
	print "\n"

	# Check that unit variance exists
	print standardized_matrix.std(axis=0)
	print "\n"