from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
import numpy as np
import sys
from time import time


def is_float(input_val):
	"""Return whether the input string can be converted to float"""
	try:
		float(input_val)
		return True
	except:
		return False


def correct_rows(p):
	duration = p[4]
	distance = p[5]
	fare_amount = p[11]
	surcharge = p[15]
	total_amount = p[16]

	if len(p) == 17 and is_float(duration) and 0 < float(duration) < 2280 and is_float(distance) and \
			0 < float(distance) < 9.41 and is_float(fare_amount) and 0 < float(fare_amount) < 32.5 and \
			0 <= is_float(surcharge) < 39.5 and is_float(total_amount)  and \
			0 < float(total_amount) < 39.5:
		return True
	return False


def to_float(input_val):
	return [float(each) for each in input_val]


def get_statistic(input_val):
	xi = input_val[0]
	yi = input_val[1]
	xi_squared = xi ** 2
	yi_squared = yi ** 2
	xiyi = xi * yi
	return np.array([xi, yi, xi_squared, yi_squared, xiyi, 1])


def train_slr(input_val, m, b):
	xi = input_val[0]
	yi = input_val[1]
	pred = m * xi + b
	cost = (pred - yi) ** 2
	gradient_m = xi * (pred - yi)
	gradient_b = pred - yi
	return np.array([gradient_m, gradient_b, cost, 1])


def train_mlr(input_val, weights, bias):
	x = input_val[0]
	y = input_val[1]
	pred = weights.dot(x) + bias
	gradient_bias = pred - y
	gradient_weights = x * gradient_bias
	cost = (pred - y) ** 2
	return gradient_weights, gradient_bias, cost, 1


if __name__ == "__main__":
	sc = SparkContext()
	input_dir = sys.argv[1]
	data = sc.textFile(input_dir)

	data = data.map(lambda x: x.split(","))
	data = data.filter(correct_rows)  # cleaning data

	'''Task1 Simple Linear Regression'''
	print("================= Task1 Simple Linear Regression ===================")
	time_start = time()
	data_slr = data.map(lambda x: (x[5], x[11])).map(lambda x: (float(x[0]), float(x[1])))
	data_slr.persist(storageLevel=StorageLevel(True, False, False, False))
	x, y, x_squared, y_squared, xy, n = data_slr.map(get_statistic).reduce(np.add)
	m = (n * xy - x * y) / (n * x_squared - x ** 2)
	b = (x_squared * y - x * xy) / (n * x_squared - x ** 2)
	print("x: {}, y : {}, x_squared: {}, y_squared: {}, xy: {}, n: {}".format(x, y, x_squared, y_squared, xy, n ))
	print("Equation of Simple Linear Regression : fare = {} * distance + {}".format(m, b))
	time_end = time()
	print("Time Cost: {}".format(time_end-time_start))

	'''Task2 Find the Parameters using Gradient Descent'''
	print("================= Task2 Find the Parameters using Gradient Descent ===================")
	num_iteration = 100
	learning_rate = 0.001
	m = 0.1
	b = 0.1
	for i in range(num_iteration):
		sample = sc.parallelize(data_slr.takeSample(True, 10000))
		gradient_m_sum, gradient_b_sum, cost, size = sample.map(lambda x: train_slr(x, m, b)).reduce(np.add)
		gradient_m = 2 * gradient_m_sum / size
		gradient_b = 2 * gradient_b_sum / size
		m = m - learning_rate * gradient_m
		b = b - learning_rate * gradient_b
		print(i, "weight: {}  bias: {}   cost: {} ".format(m, b, cost))

	'''Task3 Fit Multiple Linear Regression using Gradient Descent'''
	print("================= Task3 Fit Multiple Linear Regression using Gradient Descent ===================")
	data_mlr = data.map(lambda x: (x[4], x[5], x[11], x[15], x[16])).map(to_float)
	data_mlr = data_mlr.map(lambda x: (np.array(x[0:4]), x[4]))
	data_mlr.persist(storageLevel=StorageLevel(True, False, False, False))
	num_iteration = 100
	learning_rate = 0.001
	weights = np.full([4], 0.1)
	bias = 0.1

	old_cost = None
	for i in range(num_iteration):
		sample = sc.parallelize(data_mlr.takeSample(True, 10000))
		gradients_sum, gradient_bias, cost, n = sample.map(lambda x: train_mlr(x, weights, bias)).reduce(
			lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))
		gradients_weights = 2 * gradients_sum / n
		gradient_bias = 2 * gradient_bias / n
		weights = weights - learning_rate * gradients_weights
		bias = bias - learning_rate * gradient_bias
		if old_cost is None:
			old_cost = cost
		else:
			if cost <= old_cost:
				learning_rate *= 1.05
			else:
				learning_rate *= 0.5
			old_cost = cost
		print(i, "weights: {}  bias: {}   cost: {} ".format(weights, bias, cost))
