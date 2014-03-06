from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sk_metrics
import numpy as np
import math
import os

'''
	Given a matrix, calculate the mean and standard deviations for columns
	Standardize the matrix elements by applying the z-score formula:
	(x - mean)/std_deviation
'''
def standardize(matrix):
	preprocessed_data = preprocessing.scale(matrix)
	return preprocessed_data

def binarize(matrix):
	preprocessed_data = preprocessing.binarize(matrix)
	return preprocessed_data

def log_map(elem):
	transform_const = 0.1
	return math.log(elem + 0.1)

def log_transform(matrix):
	for i, instance in enumerate(matrix):
		matrix[i] = map(log_map, instance)
	return matrix

def init_files():
	f1 = os.getcwd() + "/spam.traintest.txt"
	f2 = os.getcwd() + "/spam.data"

	return f1, f2
'''
	Using the indicator file, this function filters out the training data & test data from the 
	raw input file
'''
def filter_data(indication_matrix, data_matrix):
	training_data = [data_matrix[i] for i, j in enumerate(indication_matrix) if j == 0]
	test_data = [data_matrix[i] for i, j in enumerate(indication_matrix) if j == 1]
	return np.asarray(training_data), np.asarray(test_data)

def extract_results(training_data, test_data):
	return training_data[:, 57:], test_data[:, 57:]

def calculate_MAE(calculated_results, actual_results):
	error = sum([((calculated_results[idx] - actual_results[idx]) ** 2) for idx in xrange(0, len(calculated_results))])

	return error/len(calculated_results)

def init_regression_model(training_matrix, training_results):
	regression_model = LogisticRegression()
	regression_model.fit(training_matrix, training_results)

	return regression_model

def predict_results(test_data, regression_model):
	return regression_model.predict(test_data)

if __name__ == '__main__':
	'''
		Begin Dataset processing 
	'''
	indicator_file, data_file = init_files()
	indication_matrix = np.genfromtxt(indicator_file)

	data_matrix = np.genfromtxt(data_file)
	training_matrix, test_matrix = filter_data(indication_matrix, data_matrix)
	training_results, test_results = extract_results(training_matrix, test_matrix)

	#training_matrix = binarize(training_matrix[:, :57])
	#test_matrix = binarize(test_matrix[:, :57])

	training_matrix = log_transform(training_matrix[:, :57])
	test_matrix = log_transform(test_matrix[:, :57])

	'''
		End of Dataset processing
	'''

	log_regresson_model = init_regression_model(training_matrix, training_results)
	my_results = log_regresson_model.predict(training_matrix)

	print sk_metrics.mean_squared_error(training_results, my_results)