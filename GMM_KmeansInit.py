"""
CS 6601 Artificial Intelligence
Part 2: Implementation of Gaussian Mixture Models and EM using K-Means initialization

This program runs the Expectation Maximization algorithm on an input image
using Gaussian Mixture Model.

In this example, the means are initialized using K-Means algorithm
"""
import numpy
from kmeans_methods import *

numClusters = 3
inputImage = Image.open("party_spock.png")

width, height = inputImage.size
pixels = numpy.asarray(inputImage)
pixels = pixels.transpose()

mean = runKMeans(inputImage, numClusters)

sigma = [1 for x in range(numClusters)]

w = [1/float(numClusters) for x in range(numClusters)] 

stats = [mean, sigma, w, inputImage]

print("\n\n======Running Expectation Maximization Algorithm=======")
print "\nInitial mean values using K-means: " + str(mean)
print "\nInitial variance values : " + str(sigma) 
print "\nInitial mixing coefficients : " + str(w)

iterations = 100
i = 0
while(True):
    stats = gaussianMixtureModel(stats[0], stats[1], stats[2], stats[3])
    i += 1
    if i == 100:
        break