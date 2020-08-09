"""
Module that contains functions useful for a Neural Network
==================================

Provides
   Activation Functions
   1. sigmoid: returns a value between 0 and 1
   2. heavside: returns 1 if input is grater than 0, 0 otherwise
   3. identity: returns x lol
   4. softmax: returns the probability distribution of the layer

   Error Functions
   1. sumOfSquares: useful for regression (continue)
   2. crossEntropy: useful for classification (discrete)

   Other Functions
   1. sign: returns the sign of a value (+1; 0; -1)
"""
import numpy as np

#getter
def getIdentity():
    if _Identity._instance is None:
        _Identity._instance = _Identity()
    return _Identity._instance

def getSigmoid():
    if _Sigmoid._instance is None:
        _Sigmoid._instance = _Sigmoid()
    return _Sigmoid._instance

def getHeavside():
    if _Heavside._instance is None:
        _Heavside._instance = _Heavside()
    return _Heavside._instance

def getSoftmax():
    if _Softmax._instance is None:
        _Softmax._instance = _Softmax()
    return _Softmax._instance

def getSoftmax_forCrossEntropy():
    if _Softmax_forCrossEntropy._instance is None:
        _Softmax_forCrossEntropy._instance = _Softmax_forCrossEntropy()
    return _Softmax_forCrossEntropy._instance

def getSumOfSquares():
    if _SumOfSquares._instance is None:
        _SumOfSquares._instance = _SumOfSquares()
    return _SumOfSquares._instance

def getCrossEntropy():
    if _CrossEntropy._instance is None:
        _CrossEntropy._instance = _CrossEntropy()
    return _CrossEntropy._instance

def getCrossEntropy_forSoftmax():
    if _CrossEntropy_forSoftmax._instance is None:
        _CrossEntropy_forSoftmax._instance = _CrossEntropy_forSoftmax()
    return _CrossEntropy_forSoftmax._instance

# Activation Functions
class ActivationFunction:
    _instance = None

    def calculate(self, x):
        pass

    def calculateDerivative(self, x):
        pass

class _Identity(ActivationFunction):
    def calculate(self, x):
        return x  # lol

    def calculateDerivative(self, x):
        return np.ones((1, x.size))

class _Sigmoid(ActivationFunction):
    maxFloat64 = np.finfo(np.float64).max
    epsFloat64 = np.finfo(np.float64).eps
    tendsToOne = 1.0/(1.0+(np.finfo(np.float32).eps/16))
    smallest = np.nextafter(0, 1)
    def calculate(self, x):
        '''return 1.0 / (1.0 + np.exp(-x))'''
        expWontOverflow = -x < np.log(self.maxFloat64)
        expWontUnderflow = -x > np.log(self.epsFloat64)
        return np.where(expWontOverflow,
               np.where(expWontUnderflow, 1.0 / (1.0 + np.exp(-x, where=expWontOverflow)), self.tendsToOne),
               self.smallest)

    def calculateDerivative(self, x):
        return self.calculate(x) * (1.0 - self.calculate(x))

class _Heavside(ActivationFunction):
    def calculate(self, x):
        return 1.0 if x > 0 else 0.0

    def calculateDerivative(self, x):
        return np.zeros(x.size)

class _Softmax(ActivationFunction):
    def calculate(self, x):
        #shiftedX = x - np.max(x)
        exp = np.exp(x)
        sum = np.sum(exp)
        return exp/sum

    def calculateDerivative(self, x):
        ret = ( 1.0 / self.calculate(x) ) - 1.0
        return ret.reshape(1, len(x))

class _Softmax_forCrossEntropy(ActivationFunction):
    def calculate(self, x):
        exp = np.exp(x)
        sum = np.sum(exp)
        return exp/sum

    def calculateDerivative(self, x):
        return np.ones(np.size(x))

#Funzioni d'errore
class ErrorFunction:
    _instance = None

    def calculate(self, y, t):
        pass

    def calculateDerivative(self, y, t):
        pass

    def areSameSize(self, y, t):
        if y.size != t.size:
            raise Exception("The arguments must have same size!")
        return True

class _SumOfSquares(ErrorFunction):
    def calculate(self, y, t):
        if self.areSameSize(y, t):
            diff = y - t
            result = np.sum(diff**2)
            return result/2.0

    def calculateDerivative(self, y, t):
        print("y and t", y, t)
        if self.areSameSize(y, t):
            return np.sum(y-t);

class _CrossEntropy(ErrorFunction):
    smallest = np.nextafter(0, 1)
    def calculate(self, y, t):
        if self.areSameSize(y, t):
            y= np.where(y==0, self.smallest, y)
            return -np.sum(t * np.log(y))

    def calculateDerivative(self, y, t):
        if self.areSameSize(y, t):
            return np.sum(-t / y)

class _CrossEntropy_forSoftmax(ErrorFunction):
    smallest = np.nextafter(0, 1)
    def calculate(self, y, t):
        if self.areSameSize(y, t):
            y= np.where(y==0, self.smallest, y)
            return -np.sum(t * np.log(y))

    def calculateDerivative(self, y, t):
        if self.areSameSize(y, t):
            ret = y - t
            return ret.reshape(1, len(y))

def sign(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))