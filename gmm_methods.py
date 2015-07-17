import math
import random
import numpy
import warnings
from PIL import Image


def gaussianMixtureModel(mean, sigma, pi, inputImage):
    """
    Run Expectation Maximization algorithm using Gaussian Mixture Model
    """
    width, height = inputImage.size
    numClusters = len(mean)
    pixels = numpy.asarray(inputImage).transpose()
    
    """**************************************
    Estimation - compute log(responsibility)
    **************************************"""
    
    logRes = numpy.zeros(shape=(width, height, numClusters), dtype=float )
    for x in range(width):
        for y in range(height) :
            for k in range(numClusters):
                Xn=pixels[x][y]
                logRes[x][y][k] = math.log(pi[k]) + logPdf(Xn, mean[k], sigma[k])
            sum=0
            for k in range(numClusters):
                if(k==0):
                    sum=logRes[x][y][k]
                else:
                    sum = logSum(sum, logRes[x][y][k])
            for k in range(numClusters):
                logRes[x][y][k]= logRes[x][y][k] - sum 
    
    """**************************************
    #2-Maximization - compute new parameters
    **************************************"""
    """
    Compute New Mean
    """
    mean = []
    for k in range(numClusters):
        numerator=0
        denominator=0
        for x in range(width):
            for y in range(height):
                numerator=numerator+(pixels[x][y] * float(math.exp(logRes[x][y][k]) ))
                denominator=denominator+float(math.exp(logRes[x][y][k]))
        
        mean.append((float(numerator))/float(denominator))
    
    """
    Compute new variance
    """
    sigma = []
    for k in range(numClusters):
        numk = 0
        denk = 0
        for x in range(width):
            for y in range(height):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    numk += (float(math.exp(logRes[x][y][k]) )) * ((pixels[x][y]-mean[k])**2)
                denk += float(math.exp(logRes[x][y][k]))
        sigma.append((float(numerator))/float(denominator))
    
    """
    Compute new mixing coefficients
    """
    w = []
    for k in range(numClusters):
        numerator=0
        for x in range(width):
            for y in range(height):
                numerator=numerator+float(math.exp(logRes[x][y][k] ))
        w.append(float(numerator)/float(pixels.size))
    
    """
    Compute log likelihood
    """
    logLikelihood_intermediate = getLogLikelihood(pi, mean, sigma, pixels)
    
    print "\nNew Mean Values : " + str(mean)
    print "\nNew Variance Values : " + str(sigma)
    print "\nNew mixing coefficients : " + str(w)
    print "\nLog Likelihood :" + str(logLikelihood_intermediate)
    
    """
    Update image with new clustering info and display it
    """
    renderImage = inputImage
    for x in range(width):
            for y in range(height):
                logRes[x][y] = numpy.exp (logRes[x][y])
                highestKIndex = numpy.argmax(logRes[x][y])
                renderImage.putpixel((x, y), (mean[highestKIndex]))             
    renderImage.show()

    return [mean, sigma, w, inputImage]


def getLogLikelihood(pi, mean, sigma, pixels):
    logLikelihood = 0
    numClusters = len(pi)
    for x in range(len(pixels)):
        for y in range(len(pixels[0])):
            sum = 0
            for k in range(numClusters):
                Xn = pixels[x][y]
                sum += (pi[k] * math.exp(logPdf(Xn, mean[k], sigma[k])))
                if sum != 0:
                    sumLog = (math.log(sum)) 
                    logLikelihood = logSum(logLikelihood, sumLog)
    return logLikelihood
    

def logPdf(x, mean, sigma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = ((-0.5)*(math.log(2*math.pi*sigma)))+((-0.5)*(((x-mean)**2)/(2*sigma)))
            return result
        except ValueError:
            return 0


def logSum(logA, logB):
    if logA == 0:
        return logB
    elif logB == 0:
        return logA
    else:
        if(logA > logB):
            return (logA) + math.log(1 + math.exp(logB - logA))
        else:
            return (logB) + math.log(1 + math.exp(logA - logB))
        

