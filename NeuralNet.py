import numpy as np
import os, struct
from array import array as pyarray
from math import *

class Node:
    def __init__(self, inputs = 1, weights = [1], threshold = 0):
        self.inputs = inputs
        self.weights = np.transpose(np.matrix(weights))
        if len(weights) != inputs:
            self.weights = np.transpose(np.ones(inputs))
        self.threshold = threshold

    def getOutput(self):
        val = sum([self.inputs[i]*self.weights[i] for i in range(len(self.inputs))]) + self.threshold >= 0
        return 1 / (1 + exp(-val))

class InputNode:
    def __init__(self, inputVal = 0):
        self.inputVal = inputVal
        self.weights = [1]
        self.threshold = 1

    def getOutput(self):
        return self.inputVal
        
class Network:
    def __init__(self, inNodes = [], hiddenNodes = [[]], outNodes = []):
        self.inNodes = inNodes
        self.hiddenNodes = hiddenNodes  #array of arrays.
        
        self.outNodes = outNodes
        self.nodes = []
        self.nodes += [inNodes] + hiddenNodes + [outNodes]
        self.layers = len(self.nodes)

    def getOutput(self):
        return [n.getOutput() for n in self.outNodes]

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

    def feedForward(self, inputVector):
        self.loadInput(inputVector)
        Z = []
        Z.append(sigma(inputVector))
        Y = np.transpose(np.matrix(inputVector))
        A = np.transpose(np.matrix(inputVector))
        for l in range(1, self.layers):
            W = self.weightMatr(l)
            b = self.bias(l)
            print('w: ',W)
            print('a: ',A)
            print('b: ',b)
            Z.append(W*A + b)
            A = sigma(Z[l])
        sigmaPrime =  - np.array(Z[l]) + np.ones(len(Z[l]))
        sigmaPrime = sigma(sigmaPrime)
        sigmaPrime = A*sigmaPrime
        return np.matrix(np.absolute(np.array(A - Y))*sigmaPrime)
            
def sigma(val):
    print(type(val))
    if type(val) != type(1):
        for i in range(len(val)):
            val[i] = 1 / (1 + exp(-val[i]))
        return val
    return 1 / (1 + exp(-val))

def createNetwork(name = "network"):
    file = open(name + ".txt", "r")
    lines = list(file.readlines())
    inNodes = [InputNode() for i in range(int(lines[0]))]
    hiddenNodes = [[Node(inputs = int(lines[k-1])) for i in range(int(lines[k]))] for k in range(1,len(lines) - 1)]
    outputNodes = [Node(inputs = int(lines[-2])) for i in range(int(lines[-1]))]
    return Network(inNodes = inNodes, hiddenNodes = hiddenNodes, outNodes = outputNodes)

def loadMnist(dataset="training", digits=np.arange(10), path="."):
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
    for i in range(len(ind)):
        images.append(np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]))
        labels.append(lbl[ind[i]])

    return images, labels

if __name__ == '__main__':
	network = createNetwork()
	i,l = loadMnist()
