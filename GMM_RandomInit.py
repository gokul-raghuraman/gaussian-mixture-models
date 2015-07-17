"""
CS 6601 Artificial Intelligence
Part 1: Implementation of Gaussian Mixture Models and EM

This program runs the Expectation Maximization algorithm on an input image
using Gaussian Mixture Model

In this example, the means are initialized randomly
"""
import numpy
from gmm_methods import *

numClusters = 3
inputImage = Image.open("party_spock.png")
width, height = inputImage.size
pixels = numpy.asarray(inputImage).transpose()

"""
Pick random values from image as initial means 
"""
mean = []
x = numpy.random.randint(0, width, size=numClusters)
y = numpy.random.randint(0, height, size=numClusters) 
for i in range(numClusters):
    mean.append(pixels[x[i]][y[i]])

"""
Pick initial variance values of 1 for each cluster
"""
sigma = [1 for x in range(numClusters)] #initial variances

"""
Pick equal mixing coefficients of 0.33 to start with
"""
w = [1/float(numClusters) for x in range(numClusters)]

print("\n======Running Expectation Maximization Algorithm=======")
print "\nRandom initial mean values from image: " + str(mean)
print "\nInitial variance values : " + str(sigma) 
print "\nInitial mixing coefficients : " + str(w)

stats = [mean, sigma, w, inputImage]


while(True):
    stats = gaussianMixtureModel(stats[0], stats[1], stats[2], stats[3])
    