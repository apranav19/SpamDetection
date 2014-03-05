import numpy as np 
import math
import os

'''
	Given a matrix, calculate the mean and standard deviations for columns
	Standardize the matrix elements by applying the z-score formula:
	(x - mean)/std_deviation
'''
def standardize(matrix):
	feature_averages = matrix.mean(axis=0)	# Unidimensional array containing means for each column
	feature_std_devs = matrix.std(axis=0)	# Unidimensional array containing std_devs for each column

	for i, row in enumerate(matrix):
		for j, col in enumerate(matrix[i]):
			standardized_val = (matrix[i, j] - feature_averages[j])/(feature_std_devs[j]) # Compute z-score
			matrix[i, j] = standardized_val # Set element to its computed z-score

	return matrix


def sigmoid(instance, weight_instance):
	denom = 1.0 + math.exp((-1 * np.dot(weight_instance.T, instance)))
	return 1/denom

def gradient_descent(learning_rate, input_matrix, results, weights):
	while convergence <= 0.01: 
		for j in xrange(0, len(weights)):
			sum_likelihood = 0
			for i in xrange(0, len(input_matrix)):
				sum_likelihood += (sigmoid(input_matrix[i], weights) - results[i]) * input_matrix[i][j]

			weights[j] = weights[j] - (learning_rate * sum_likelihood)
			convergence = weights[j] - weights[j-1]
	
	return weights

def fill_bias_features(input_matrix):
	return np.insert(input_matrix, 0, 1, axis=1)


def init_files():
	f1 = os.getcwd() + "/spam.traintest.txt"
	f2 = os.getcwd() + "/spam.data"

	return f1, f2

def filter_data(indication_matrix, data_matrix):
	training_data = [data_matrix[i] for i, j in enumerate(indication_matrix) if j == 0]
	test_data = [data_matrix[i] for i, j in enumerate(indication_matrix) if j == 1]

	return np.asarray(training_data), np.asarray(test_data)

def extract_results(training_data, test_data):
	return training_data[:, 57:], test_data[:, 57:]


if __name__ == '__main__':
	indicator_file, data_file = init_files()

	indication_matrix = np.genfromtxt(indicator_file)

	data_matrix = np.genfromtxt(data_file)
	training_matrix, test_matrix = filter_data(indication_matrix, data_matrix)

	training_results, test_results = extract_results(training_matrix, test_matrix)

	training_matrix = training_matrix[:, :57]
	test_matrix = test_matrix[:, :57]

	print "Dimension of training: " + str(training_results.shape) + " \n"
	print " Dimension of test: " + str(test_results.shape)