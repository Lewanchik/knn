from numpy import * 
import operator
import sklearn.metrics.pairwise as slp
from sklearn import datasets
from skimage import exposure
import numpy as np
import sklearn
import imutils
import cv2
from PIL import Image, ImageDraw
from mnist import MNIST

#firtly need to create Directory 'samples' with all files from  http://yann.lecun.com/exdb/mnist/ and rename it from .idx3-ubyte to -idx3-ubyte

mndata = MNIST('samples')#get data from MNIST files
x_test, y_test = mndata.load_testing()#10000 images and labels
x_train, y_train = mndata.load_training()#60000 images and labels
#creates numpy arrays from data
x_train = np.array(x_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_train = np.array(y_train)

def makeLabels(labels):#make dict of labels and iterator EXAMPLE :{1:label1, 2:label2,...}
	newLabels = {}
	for i in range(len(labels)):
		newLabels.update({i : labels[i]})
	return newLabels

def maxImage(image):#create smaller image with filter
	width = 28#width of start image
	shape = 2#size of filter
	image = image.reshape((width, width)).astype("uint8")
	newMatrix = np.array([])
	maxValue = 0
	i = 0
	while i < width:
		j = 0
		while j < width:
			maxValue = image[i][j]
			for k in range(i, i + shape):
				for l in range(j, j + shape):
					if (maxValue < image[k][l]):
						maxValue = image[k][l]
			newMatrix = np.append(newMatrix, [maxValue], axis = 0)
			j = j + shape
		i = i + shape
	return newMatrix

def	maxDataSet(dataSet):#create dataset with smaller image
	dataSetSize = len(dataSet)
	imageSize = 196#size of new image = (size of old image)/(shape**2) SHAPE IT IS VALUE FROM maxImage FUNCTION
	newDataSet = np.zeros(shape=(dataSetSize, imageSize))
	for i in range(dataSetSize):
		newDataSet[i] = (maxImage(dataSet[i])).ravel()
	return newDataSet

def kNNCOSINE(x, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	distances = np.zeros(shape=(1, dataSetSize)).ravel()
	for i in range (dataSetSize):
		distances[i] = 1 - np.sum(dataSet[i] * x) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(dataSet[i]**2)))
	sortedDistances = distances.argsort()
	classCount = {}
	classDist = {}
	finalDist = {}
	for i in arange(k):         
		votelabel = labels[sortedDistances[i]]
		classCount[votelabel] = classCount.get(votelabel,0) + 1
		if (votelabel in classDist.keys()):
			classDist[votelabel] = classDist[votelabel] + distances[sortedDistances[i]]
		else:
			classDist[votelabel] = distances[sortedDistances[i]]
	for i in classDist.keys():
		finalDist[i] = classDist[i]/classCount[i]
	sortedFinal = sorted(finalDist.items(), key=operator.itemgetter(1), reverse=False) 
	return sortedFinal[0][0] 

def kNNEUCLIDEAN(x, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diff = tile(x, (dataSetSize,1)) - dataSet
	sqDiff = diff ** 2
	sqDistances = sqDiff.sum(axis=1)
	distances = np.sqrt(sqDistances)
	sortedDistances = distances.argsort()
	classCount = {}
	classDist = {}
	finalDist = {}
	for i in arange(k):         
		votelabel = labels[sortedDistances[i]]
		classCount[votelabel] = classCount.get(votelabel,0) + 1
		if (votelabel in classDist.keys()):
			classDist[votelabel] = classDist[votelabel] + distances[sortedDistances[i]]
		else:
			classDist[votelabel] = distances[sortedDistances[i]]
	for i in classDist.keys():
		finalDist[i] = classDist[i]/classCount[i]
	sortedFinal = sorted(finalDist.items(), key=operator.itemgetter(1), reverse=False) 
	return sortedFinal[0][0] 

def kNNMANHATTAN(x, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diff = tile(x, (dataSetSize,1)) - dataSet
	sqDiff = np.abs(diff)
	distances = sqDiff.sum(axis=1)
	sortedDistances = distances.argsort()
	classCount = {}
	classDist = {}
	finalDist = {}
	for i in arange(k):         
		votelabel = labels[sortedDistances[i]]
		classCount[votelabel] = classCount.get(votelabel,0) + 1
		if (votelabel in classDist.keys()):
			classDist[votelabel] = classDist[votelabel] + distances[sortedDistances[i]]
		else:
			classDist[votelabel] = distances[sortedDistances[i]]
	for i in classDist.keys():
		finalDist[i] = classDist[i]/classCount[i]
	sortedFinal = sorted(finalDist.items(), key=operator.itemgetter(1), reverse=False) 
	return sortedFinal[0][0] 

error = 0
newLabels = makeLabels(y_train)

for i in range(1000):#cycle for test first 1000  of testing images
	if (kNNEUCLIDEAN(x_test[i], x_train, newLabels, 2) != y_test[i]):
		error = error + 1

print("error2 = ", error)#print error rate




