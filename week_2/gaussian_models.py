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


# Generate a random dataset
dataset_size = 4
dimension = 2
dataset_A = np.random.rand(dataset_size, dimension)*(2,1) + (5, 5)
dataset_B = np.random.rand(dataset_size, dimension)*(1,5) + (2, 1)

# Plot the datasets
plt.title('Plot of Data Sets A and B')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
legend(['Set A', 'Set B'])
plt.scatter(x = dataset_A.T[0], y = dataset_A.T[1], color="#CC0000", edgecolors='#220000')
plt.scatter(x = dataset_B.T[0], y = dataset_B.T[1], color="#007700", marker="^", edgecolors='#002200')

def mu(dataset):
	return sum(dataset) / len(dataset)

def variance(x, mu):
	total = 0.0
	for i in range(len(x)):
		total += math.pow(x[i] - mu, 2)
	return total

def covariance(dataset):
	mu_x = mu(dataset.T[0])
	mu_y = mu(dataset.T[1])
	size = len(dataset)
	deviation = dataset - np.dot(np.ones((size, size)), dataset) / len(dataset)
	return np.dot(deviation.T, deviation) / len(dataset)

def single_cov(dataset):
	mu_x = mu(dataset.T[0])
	mu_y = mu(dataset.T[1])
	return sum([(x - mu_x) * (y - mu_y) for [x, y] in dataset]) / (len(dataset) - 1)

def norm_pdf_multivariate(xy, mu, sigma):
	size = len(xy)
	print(size)
	print(len(mu))
	print(sigma.shape)
	if size == len(mu) and (size, size) == sigma.shape:
		det = np.linalg.det(sigma)
		if det == 0:
			raise NameError("The covariance matrix can't be singular")
		norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
		x_mu = np.matrix(xy - mu)

		result = math.pow(math.e, -0.5 * (x_mu * np.linalg.inv(sigma) * x_mu.T))
		return norm_const * result
	else:
		raise NameError("The dimensions of the input don't match")

def norm_pdf_multivariate2(xy, mu, sigma):
    size = len(xy)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(xy - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


if(mu(dataset_A[0]) == np.mean(dataset_A[0])):
	print("Calculated mu() = np.mean()")

# print(np.cov(dataset_A.T))
# print(covariance(dataset_A))
print(norm_pdf_multivariate(dataset_A.T, mu(dataset_A), covariance(dataset_A)))

plt.show()

