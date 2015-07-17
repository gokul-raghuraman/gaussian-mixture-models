import math
import random
import numpy
from PIL import Image
from gmm_methods import *

        
def getClosestVal(x, mean):
    return mean[(numpy.abs(mean - x)).argmin()]

def getKMeans(inputImage, mean):
    width, height = inputImage.size
    pixels = numpy.asarray(inputImage)
    pixels = pixels.transpose()
    pixCoord = numpy.empty(shape=(width,height,2))
    
    for x in range(width):
        for y in range(height):
            
            pixCoord[x][y][0]=pixels[x][y]
            Xn=pixCoord[x][y][0]
            closest_index= numpy.abs(mean-Xn).argmin()
            pixCoord[x][y][1]=closest_index
            
    for k in range(len(mean)):
        current_mean_array = []
        for x in range(width):
            for y in range(height):
                if (pixCoord[x][y][1]==k):
                    current_mean_array.append(pixCoord[x][y][0])
        current_mean_array=numpy.asarray(current_mean_array)
        mean[k] = numpy.mean(current_mean_array) 
    
    return mean   
                
def runKMeans(inputImage, numClusters, iterations=5):
    inputImage = Image.open("party_spock.png")

    width, height = inputImage.size
    pixels = numpy.asarray(inputImage)
    pixels = pixels.transpose()

    mean = []
    x = numpy.random.randint(0, width, size = numClusters)
    y = numpy.random.randint(0, height, size = numClusters) 
    for i in range(numClusters):
        mean.append(pixels[x[i]][y[i]])
    print("\n======Running K-Means Algorithm=======")
    print "Randomly selecting initial means = " + str(mean)
    iterations = 5
    i = 0
    while (True):
        mean = getKMeans(inputImage, mean)
        print "\nIteration " + str(i + 1) + " : means = " + str(mean)
        i += 1
        if i == iterations:
            break
    return mean
