import numpy as np
import os, struct
from array import array as pyarray
from math import *
import random

errorCollect = []

class Node:
    def __init__(self, inputs, weights = None, threshold = 0):
        self.inputs = inputs
        if weights != None:
            self.weights = np.array(weights, dtype=float)
        elif weights == None:
            self.weights = np.array([np.random.uniform(-(1/len(inputs)), (1/len(inputs))) for i in range(len(inputs))], dtype=float)
        self.threshold = threshold

    def getOutput(self, inputVals):
        val = sum([inputVals[i]*self.weights[i] for i in range(len(self.inputs))]) + self.threshold
        return 1 / (1 + exp(-val))

class InputNode:
    def __init__(self, inputVal = 0):
        self.inputVal = inputVal
        self.weights = [1]
        self.threshold = 1

    def getOutput(self):
        return 1 / (1 + exp(-self.inputVal))
        
class Network:
    def __init__(self, inNodes = [], hiddenNodes = [[]], outNodes = []):
        self.inNodes = inNodes
        self.hiddenNodes = hiddenNodes  #array of arrays.
        
        self.outNodes = outNodes
        self.nodes = []
        self.nodes += [inNodes] + hiddenNodes + [outNodes]
        self.layers = len(self.nodes)

    def getOutput(self, inputVal):
        self.loadInput(inputVal)
        current = inputVal
        for l in range(1,self.layers):
            nextVals = []
            for n in self.nodes[l]:
                nextVals.append(n.getOutput(current))
            current = nextVals
        return current.index(max(current))

    def loadInput(self, inputVector):
        for b in range(len(inputVector)):
            self.inNodes[b].inputVal = inputVector[b]

    def weightMatr(self, layer):
        length = len(self.nodes[layer - 1])
        height = len(self.nodes[layer])
        W = np.matrix(np.zeros((height, length)))
        for j in range(0, height):
            W[j,:] = self.nodes[layer][j].weights
        return W

    def bias(self, layer):
        L = np.zeros(len(self.nodes[layer]))
        for l in range(0,len(L)):
            L[l] = self.nodes[layer][l].threshold
        return np.transpose(np.matrix(L))

    def feedForward(self, inputVector, label):
        '''
        Verified to work as expected
        '''
        self.loadInput(inputVector)
        Z = []
        Z.append(inputVector)
        Y = np.zeros(len(self.nodes[-1]))
        Y[label] = 1
        A = np.transpose(np.matrix(inputVector))
        for l in range(1, self.layers):
            W = self.weightMatr(l)
            b = self.bias(l)
            Z.append(W*A + b)
            A = np.array(sigma(Z[l]))
        A = A.transpose()
        #print("A: ", A)
        #print("Y: ", Y)
        #print("Z[l]: ", Z[l])
        finalError = np.absolute(A - Y).transpose()*sigmaPrime(Z[l])
        return (Z, finalError)

    def backProp(self, finalError, Z):
        '''
        Verified: works as expected
        '''
        Delta = []
        Delta.append(finalError)
        for l in range(self.layers - 2, -1, -1):
            W = self.weightMatr(l+1)
            Delta.append(np.array(W.transpose()*Delta[-1])*sigmaPrime(Z[l]))
        return Delta[::-1]

    def gradDescent(self, learnRate, Delta, Z):
        '''
        Verified: works as expected
        '''
        Z[0] = np.transpose(np.matrix(Z[0]))
        for l in range(1,self.layers):
            A = np.transpose(sigma(Z[l-1]))
            delt = np.matrix(Delta[l], dtype=float)
            gradMatr = np.transpose(delt*A)
            for i,n in enumerate(self.nodes[l]):
                for j,w in enumerate(self.nodes[l][i].weights):
                    self.nodes[l][i].weights[j] -= learnRate*gradMatr[j,i]
                self.nodes[l][i].threshold -= learnRate*delt[i]

    def learn(self, inputVals, inputLabels, alpha):
        counter = 1
        for inputVal, inputLabel in zip(inputVals, inputLabels):
            Z, finalError = self.feedForward(inputVal, inputLabel)
            delta = self.backProp(finalError, Z)
            errorCollect.append(delta)
            self.gradDescent(alpha, delta, Z)
            print(counter)
            counter += 1

            
def sigma(val):
    if type(val) != type(1):
        val = 1 / (1 + np.exp(-val))
        return val
    return 1 / (1 + exp(-val))

def sigmaPrime(val):
    return np.array(sigma(val))*np.array(np.matrix(np.ones(len(val))).transpose() - sigma(val))
    

def createNetwork(name = "network", weightsRand = True):
    file = open(name + ".txt", "r")
    lines = list(file.readlines())
    lines = [l for l in lines if l[0] != '#']
    inNodes = [InputNode() for i in range(int(lines[0]))]
    hiddenNodes = []
    if weightsRand:
                hiddenNodes.append([Node(inNodes) for i in range(int(lines[1]))])
    else:
        hiddenNodes.append([Node(inNodes, weights = [1 for i in range(len(inNodes))]) for i in range(int(lines[1]))])
    for k in range(0, len(lines)-3):
        if weightsRand:
            hiddenNodes.append([Node(hiddenNodes[k]) for i in range(int(lines[k+1]))])
        else:
            hiddenNodes.append([Node(hiddenNodes[k], weights = [1 for i in range(len(hiddenNodes[k]))]) for i in range(int(lines[k+1]))])
    if weightsRand:
        outputNodes = [Node(hiddenNodes[-1]) for i in range(int(lines[-1]))]
    else:
        outputNodes = [Node(hiddenNodes[-1], weights = [1 for i in range(len(hiddenNodes[-1]))]) for i in range(int(lines[-1]))]
    return Network(inNodes = inNodes, hiddenNodes = hiddenNodes, outNodes = outputNodes)

def loadMnist(dataset="training", num = 6000, digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays
    Adapted from: http://g.sweyla.com/blog/2012/mnist-numpy/
        which was adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = 'train-images.idx3-ubyte'
        fname_lbl = 'train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_img = 't10k-images-idx3-ubyte'
        fname_lbl = 't10k-labels-idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = list(flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = list(fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = []
    labels = []
    for i in range(min(len(ind), num)):
        images.append(np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]))
        labels.append(lbl[ind[i]])
    return images, labels

def normalize(datasets):
    squaredsum = 0
    mean = 0
    N = 0
    for data in datasets:
        squaredsum += sum([x^2 for x in data])
        N += len(data)
        mean += np.mean(data)
    stdev = sqrt(squaredsum / N)
    mean = mean / len(datasets)
    
    return [(data - mean)/stdev for data in datasets]

if __name__ == '__main__':
	network = createNetwork()
	print('network initialized')
	i,l = loadMnist(num = 50)
	print('training data loaded')
	#i = normalize(i)
	i = [image/255 for image in i]
	print('data normalized')
	network.learn(i,l,.1)
	print('network taught')
