'''
@author ysbecca
Generating a random dataset using python, modeling the dataset using Gaussian distributions, 
and plotting the data and Gaussian models in 3d using Python and library numpy.

'''
import math
import numpy as np

from ys_inputs import inputs

print("Hello Perceptron.")

# Total number of inputs, including BIAS
M = len(inputs[0])
data_size = len(inputs)
print(M)
print(data_size)

# Maximum iterations to try:
T = 100
learning_rate = 0.3
targets = [0 for x in range(data_size)]

# Save all the target outputs into targets[] and add the bias
for x in range(data_size):
	targets[x] = inputs[x][-1]
	inputs[x][-1] = -1.0

# Set all the weights to small random numbers - (M * data_size) weights total
weights = np.random.uniform(-1, 1, M)
activation = [0.0 for x in range(data_size)]


def training(inputs, weights, activation, learning_rate, M, targets, data_size):
	# Compute activation.
	compute_activation(inputs, weights, activation, data_size, M)

	for v in range(data_size): # For each input vector
		for i in range(M): # Update all M weights
			change = learning_rate * (activation[v] - targets[v]) * inputs[v][i]
			# Update each of the weights individually
			weights[i] = weights[i] - change

def compute_activation(inputs, weights, activation, data_size, M):
	for v in range(data_size): # For each input vector
		activation[v] = 0
		
		# Compute the activation of the ONE neuron by summing over all M+1 weights
		for i in range(M):
			activation[v] += weights[i] * inputs[v][i]
		# Now simplify the sum to show whether the neuron should fire or not
		if activation[v] > 0:
			activation[v] = 1
		else:
			activation[v] = 0


def verify(activation, targets, data_size):
	correct = 0
	for x in range(data_size):
		if(activation[x] == targets[x]):
			correct += 1
	return 100 * float(correct) / float(data_size)


for i in range(T):
	training(inputs, weights, activation, learning_rate, M, targets, data_size)
	# print("Training round " + str(i + 1) + ":")
	compute_activation(inputs, weights, activation, data_size, M)
	# print(activation)
	if(activation == targets):
		print("Matched target output! Took " + str(i + 1) + " iterations.")
		break

print(str(verify(activation, targets, data_size)) + "% correct")
print("Ending program.")

