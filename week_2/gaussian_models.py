'''
@author ysbecca
Generating a random dataset using python, modeling the dataset using Gaussian distributions, 
and plotting the data and Gaussian models in 3d using Python and library numpy.

'''
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import title, legend


print("Hello Numpy.")

def find_mu(dataset):
	size = float(len(dataset))
	total = 0.0
	for x in dataset:
		total = total + x

	return total / size


# Generate a random dataset
dataset_size = 10
dimension = 2

dataset_A = 10*np.random.rand(dataset_size, dimension)*(2,1) + (1,3)
dataset_B = 10*np.random.rand(dataset_size, dimension)*(1,2) + (-0.5, -3)


# Plot the datasets
plt.title('Plot of DataSets A and B')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
legend(['Set A', 'Set B'])

plt.plot(dataset_A, 'ro')
plt.plot(dataset_B, 'bo')
plt.show()

# print("Covariance:")
# print(np.cov(dataset_A))
# print("Determinant:")
# print(np.linalg.det(np.cov(dataset_A))*math.pow(10, 135))


def gaussian(xy, mu, sigma):
	pi = 3.14159265359
	size = len(xy)
	det = np.linalg.det(sigma)*math.pow(10, 135)
	if det == 0:
		raise NameError("The determinant is 0.")

	fraction = 1.0 / (math.pow(2 * pi, float(size) / 2.0) * math.pow(det, 0.5))
	exp = -0.5 * (x_mu * matrix(xy - mu).T * sigma.I * matrix(xy - mu))
	return fraction * math.pow(math.e, exp)


print(gaussian(dataset_A[0], find_mu(dataset_A), np.cov(dataset_A)))








