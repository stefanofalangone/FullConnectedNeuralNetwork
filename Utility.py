"""
Module that contains a miscellaneous functions
==================================

Provides
   1. extractTsAndVs: returns a 2x[] matrix containing training set and validation set labels
   2. normalize: accept as argument a matrix and returns the same matrix which every value is between 0 and 1
   3. getLabelVector: accept as argument a label in [0; 9] and returns a numpyArray of all zeros except a 1 in position label
   4. getInputAsMonodimensional: converts an image (as matrix) into a monodimensional array
   5. numberOfIterations: given the start, end and step of a set (Training/Test/Validation) and returns the number of iterations
"""
import numpy as np

classNumber = 10

def setClassNumber(numberOfClasses):
    global classNumber
    classNumber = numberOfClasses

def extractTsAndVs(Y, valPercent=0.2):
    global classNumber
    labels = np.unique(Y)
    ind_T = []
    ind_V = []
    index = [ [] for x in range(classNumber) ]
    for i in range(Y.size):
        index[Y[i]].append(i)
    for i in range(classNumber):
        N = len(index[i])
        Nval = int(np.trunc(valPercent*N))
        validationRow = index[i][0:Nval]
        trainingRow = index[i][Nval:]
        ind_V.append(validationRow)
        ind_T.append(trainingRow)

    return [ind_T, ind_V]

def normalize(x, mmin=0.0, mmax=255.0):
    x = (x - mmin )/(mmax - mmin + 10**(-6))
    return x

def shufflePairedSet(firstSet, secondSet):
    z = list(zip(firstSet, secondSet))
    np.random.shuffle(z)
    return zip(*z)

def getLabelVector(lab):
    global classNumber
    labelVec = np.zeros(classNumber)
    labelVec[lab] = 1.0
    return labelVec

def getInputAsMonodimensional(input):
    monoDimInput = np.concatenate(input, axis=0)
    monoDimInput = input.reshape(1, len(monoDimInput))
    return monoDimInput

def numberOfIterations(fromIteration, toIteration, stepIteration):
    return (int)(np.ceil((toIteration-fromIteration)/stepIteration))