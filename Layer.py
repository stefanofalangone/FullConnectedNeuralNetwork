import random
import numpy as np

class Layer:
    weights = []
    output = []
    a = []
    b = []
    activationFunction = None
    neuronsNumber = 0
    delta_k = []
    derivate = []
    previousDerivate = []
    updateValue = []
    previousWeightUpdate = []

    def __init__(self, neuronsNumber, activationFunction, b):
        self.weights = []
        self.derivate = []
        self.previousDerivate = []
        self.updateValue = []
        self.previousWeightUpdate = []
        self.activationFunction = activationFunction
        self.neuronsNumber = neuronsNumber
        self.b = np.asarray(b).reshape((1, len(b)))
        self.output = np.zeros(neuronsNumber)
        self.a = np.zeros(neuronsNumber)
        self.delta_k = np.zeros(neuronsNumber)

    def createWeightsMatrix(self, numberOfWeightsForNeuron):
        self.weights = np.zeros((self.neuronsNumber, numberOfWeightsForNeuron + 1))
        self.derivate = np.zeros((self.neuronsNumber, numberOfWeightsForNeuron + 1))
        self.previousDerivate = np.zeros((self.neuronsNumber, numberOfWeightsForNeuron + 1))
        self.updateValue = 0.1 * np.ones((self.neuronsNumber, numberOfWeightsForNeuron + 1))
        self.previousWeightUpdate = 0.0 * np.ones((self.neuronsNumber, numberOfWeightsForNeuron + 1))

        for r in range(self.neuronsNumber):
            for c in range(numberOfWeightsForNeuron):
                weight = random.uniform(0.05, 0.99)
                if random.choice([True, False]):
                    weight *= -1
                self.weights[r][c] = weight
            self.weights[r][numberOfWeightsForNeuron] = self.b[0][r]

    def activate(self, x):
        self.a = None
        self.output = None
        x_b = np.append(x, 1.)
        self.a = np.asarray(np.dot(x_b, self.weights.T))
        self.output = self.activationFunction.calculate(self.a)
        return self.output

    def calculateDerivate(self, z):
        z = np.append(z, 1.)
        z = z.reshape(1, np.size(z))
        dot = np.dot(self.delta_k.T, z)
        self.derivate = dot
        return self.derivate

    def getNeuronsNumber(self):
        return self.neuronsNumber

    def getActivationFunction(self):
        return self.activationFunction

    def getWeights(self):
        return self.weights

    def addBiasToWeights(self):
        self.weights = np.append(self.weights, self.b, axis=1)