import numpy as np
import matplotlib.pyplot as plt
import StopPredicate as sp
import Utility as myU
from Function import sign
from Layer import Layer
from evaluationMetrics import getEvaluationMetrics

"""
IMPORTANT: Every array and matrix are treated as numpy array
"""

# class Net
class Net:
    layers = []
    errorFunction = None
    trainCumulativeErrorForEachEpoch = []
    validationCumulativeErrorForEachEpoch = []
    etaPlus = 1.6
    etaMin = 0.6
    def __init__(self, layersNumber, neuronsForLayer, activationFunctions, errorFunction, b):
        """
        :param layersNumber: number of layers
        :param neuronsForLayer: array of numbers
        :param activationFunctions: array of functions
        :param errorFunction:
        :param b: matrix of numbers
        """
        self.trainCumulativeErrorForEachEpoch = []
        self.validationCumulativeErrorForEachEpoch = []
        self.layers = []
        self.errorFunction = errorFunction
        myU.setClassNumber(neuronsForLayer[layersNumber-1])
        for i in range(layersNumber):
            b[i] = np.ones(len(b[i]))
            self.layers.append(Layer(neuronsForLayer[i], activationFunctions[i], b[i]))

    def build(self, X_train, Y_train, epochs=200, trainFrom=0, trainTo=5000, trainStep=1, minibatchNumber=10, validationPercentage=20):
        self.addWeightsToNeurons(X_train)
        return self.train(X_train, Y_train, epochs, trainFrom, trainTo, trainStep, minibatchNumber, validationPercentage)

    def train(self, X_train, Y_train, epochs=200, trainFrom=0, trainTo=10000, trainStep=1, minibatchNumber=10, validationPercentage=20):
        # Take 20% of training set and recalculate trainFrom
        rangeValidation = int(((trainTo - trainFrom) * validationPercentage)/100)
        validationFrom, validationTo,  = trainFrom, trainFrom + rangeValidation
        validationStep, trainFrom = trainStep,  validationTo
        sizeTrainingSet = myU.numberOfIterations(trainFrom, trainTo, trainStep)
        
        minibatchSize = np.floor(sizeTrainingSet/minibatchNumber)
        # Declaration of stopping criteria objects and boolean variables
        gl, pqDisjoint, pqOverlap = sp.getGl(1.0), sp.getPq_disjoint(strip=5, alpha=0.5), sp.getPq_overlap(strip=5, alpha=0.5)
        glEarlyStop, pqDisjointEarlyStop, pqOverlapEarlyStop = False, False, False

        labels_count = np.zeros(myU.classNumber)
        e = 0
        while e<epochs and (not(glEarlyStop) or not(pqDisjointEarlyStop) or not(pqOverlapEarlyStop)):
            print("epoch ", e, "/", epochs)
            #shuffle randomly the training set
            X_train, Y_train = myU.shufflePairedSet(X_train, Y_train)

            minibatchElementsCounter = 0 # number of elements considered, when a threshold is reached (e.g all the minibatch has been analyzed), update
            trainLocalCumulativeError = 0
            for i in range(trainFrom, trainTo, trainStep):
                input = myU.getInputAsMonodimensional(X_train[i])
                label = myU.getLabelVector(Y_train[i])
                labels_count[Y_train[i]] += 1.0
                # calculate z vector for each layer
                self.forwardPropagation(X_train[i])
                # calculate delta vector for each layer
                self.backwardPropagation(X_train[i], label)
                # for each neuron inside a layer calculate of weights derivative of error function and sum to the previous derivative (useful for updating weights)
                self.sumDerivative(input)
                minibatchElementsCounter+=1
                # update weights if it's reached the end of batch
                if minibatchElementsCounter >= minibatchSize:
                    self.update()
                    minibatchElementsCounter=0
                # calculate error on single sample
                result = self.layers[len(self.layers) - 1].output
                trainError = self.errorFunction.calculate(result, label)
                # accumulate errors on singles samples
                trainLocalCumulativeError = trainLocalCumulativeError + trainError
            # evaluate error on the whole training set
            self.trainCumulativeErrorForEachEpoch.append(trainLocalCumulativeError/sizeTrainingSet) #used to plot error
            # evaluate error on the whole validation set
            self.calculateErrorOnValidationSet(validationFrom, validationTo, validationStep, X_train, Y_train)
            validationLocalCumulativeError = self.validationCumulativeErrorForEachEpoch[-1]
            """Definition of stop criteria"""
            if not(glEarlyStop):
                if validationLocalCumulativeError < gl.getE_opt():
                    self.remeberThisNet(gl, e)
                glEarlyStop = gl.shouldEarlyStop(validationLocalCumulativeError, e)
                #if(glEarlyStop): print("GL stops at ", e)
            if not(pqDisjointEarlyStop):
                if validationLocalCumulativeError < pqDisjoint.getE_opt():
                    self.remeberThisNet(pqDisjoint, e)
                pqDisjointEarlyStop = pqDisjoint.shouldEarlyStop(trainLocalCumulativeError, validationLocalCumulativeError, e)
                #if(pqDisjointEarlyStop): print("PQ_disjoint stops at ", e)
            if not(pqOverlapEarlyStop):
                if validationLocalCumulativeError < pqOverlap.getE_opt():
                    self.remeberThisNet(pqOverlap, e)
                pqOverlapEarlyStop = pqOverlap.shouldEarlyStop(trainLocalCumulativeError, validationLocalCumulativeError, e)
                #if(pqOverlapEarlyStop): print("PQ_overlap stops at ", e)
            e += 1

        lastEndingCriteria = np.argmin(np.asarray([gl.getE_opt(), pqDisjoint.getE_opt(), pqOverlap.getE_opt()]))
        lastEndingCriteria = [gl, pqDisjoint, pqOverlap][lastEndingCriteria]
        self.updateNetThroughBestWeights(lastEndingCriteria)

        return [gl, pqDisjoint, pqOverlap]

    def test(self, X_test, Y_test, testFrom=0, testTo=1000, testStep=1, monitorEveryTest=False):
        truePositives = np.zeros(myU.classNumber)
        falseNegatives = np.zeros(myU.classNumber)
        falsePositives = np.zeros(myU.classNumber)
        for i in range(testFrom, testTo, testStep):
            result = self.vectorialPrediction(X_test[i])
            interpreted = np.argmax(result)
            if monitorEveryTest:
                print("Test su Img ", i, "(", Y_test[i], "): ", result)
                print("..expected, ", Y_test[i], "; interpreted as ", interpreted, " with ", result[interpreted], "\n")
            if interpreted != Y_test[i]:
                if monitorEveryTest:
                    print("while P(", Y_test[i], ") was ", result[Y_test[i]])
                falseNegatives[Y_test[i]] += 1.
                falsePositives[interpreted] += 1.
            else:
                truePositives[Y_test[i]] += 1.
        print("number of correct predictions(TP) ", np.sum(truePositives), " --> ", truePositives)
        print("number of Missed predictions(FN) ", np.sum(falseNegatives), " --> ", falseNegatives)
        print("number of Mismatched predictions(FP) ", np.sum(falsePositives), " --> ", falsePositives)
        testSize = myU.numberOfIterations(testFrom, testTo, testStep)
        evaluationMetrics = getEvaluationMetrics(truePositives, falsePositives, falseNegatives, testSize)
        print("testSize ", testSize)
        print("global guessRate ", evaluationMetrics.getGuessRate())
        print("Accuracy: ", evaluationMetrics.getAccuracy())
        print("Precision: ", evaluationMetrics.getPrecision())
        print("Recall: ", evaluationMetrics.getRecall())
        print("F-Score: ", evaluationMetrics.getFScore())
        print("AVG Accuracy: ", evaluationMetrics.getAVGAccuracy())
        print("Micro Precision: ", evaluationMetrics.getMicroPrecision())
        print("Micro Recall: ", evaluationMetrics.getMicroRecall())
        print("Micro F-Score: ", evaluationMetrics.getMicroFScore())
        #self.plotError()
        return [testSize, evaluationMetrics.getGuessRate(), evaluationMetrics.getAVGAccuracy(), evaluationMetrics.getMicroFScore()]

    def evaluateStoppingCriteria(self, X_test, Y_test, stoppingCriteria, testFrom=0, testTo=1000, testStep=1):
        self.updateNetThroughBestWeights(stoppingCriteria)
        resultList = self.test(X_test, Y_test, testFrom, testTo, testStep)
        resultList.append(stoppingCriteria.getStopEpoch())
        return resultList

    def addWeightsToNeurons(self, X_train):
        numberOfWeightsForNeuron = 0
        for i in range(len(self.layers)):
            if i == 0:
                numberOfWeightsForNeuron = len(X_train[0]) * len(X_train[0][0])
            else:
                numberOfWeightsForNeuron = self.layers[i - 1].getNeuronsNumber()
            self.layers[i].createWeightsMatrix(numberOfWeightsForNeuron)

    def forwardPropagation(self, x):
        x = np.concatenate(x, axis=0)  # convert x to monodimensional array
        self.layers[0].activate(x)
        for i in range(1, len(self.layers)):
            input = self.layers[i-1].output
            self.layers[i].activate(input)

    def backwardPropagation(self, x, label):
        #last layer delta is calculated here and memorized in the layers
        lastLayer = self.layers[len(self.layers) - 1]
        lastLayer.delta_k = np.asarray(self.errorFunction.calculateDerivative(lastLayer.output, label)) *lastLayer.activationFunction.calculateDerivative(lastLayer.a)
        for i in range( len(self.layers) -2 , -1, -1):
            nextLayer = self.layers[i + 1]
            weights_without_bias = np.delete(nextLayer.weights, np.size(nextLayer.weights[0]) - 1, 1)
            self.layers[i].delta_k = np.dot( nextLayer.delta_k, weights_without_bias )  * self.layers[i].activationFunction.calculateDerivative(self.layers[i].a)
        return self

    def sumDerivative(self, input):
        for j in range(len(self.layers)):
            old_derivate = self.layers[j].derivate
            self.layers[j].derivate = self.layers[j].calculateDerivate(input) + old_derivate
            input = self.layers[j].output

    def update(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.updateValue, updateWeight, layer.previousDerivate = self.calculateUpdateValue(layer.derivate, layer.previousDerivate, layer.updateValue, layer.previousWeightUpdate)
            layer.weights = layer.weights + updateWeight
            layer.previousWeightUpdate = updateWeight
            layer.derivate = np.zeros(np.size(layer))
        return self

    def calculateErrorOnValidationSet(self, validationFrom, validationTo, validationStep, X_train, Y_train):
        validationLocalCumulativeError = 0
        sizeValidationSet = myU.numberOfIterations(validationFrom, validationTo, validationStep)
        for k in range(validationFrom, validationTo, validationStep):
            predicted = self.vectorialPrediction(X_train[k])
            label = myU.getLabelVector(Y_train[k])
            validationError = self.errorFunction.calculate(predicted, label)
            validationLocalCumulativeError = validationLocalCumulativeError + validationError
        self.validationCumulativeErrorForEachEpoch.append(validationLocalCumulativeError/sizeValidationSet)

    def calculateUpdateValue(self, currentDerivateError, previousDerivateError, updateValue, previousUpdateWeight, updateMin = 1e-6 , updateMax = 1.0 ):
        currUpdateValue = np.zeros((len(currentDerivateError), len(currentDerivateError[0])))
        currUpdateWeight = np.zeros((len(currentDerivateError), len(currentDerivateError[0])))
        for i in range(len(currentDerivateError)):
            for j in range(len(currentDerivateError[i])):
                if(currentDerivateError[i][j] * previousDerivateError[i][j] > 0):
                    currUpdateValue[i][j] = min(updateValue[i][j] * self.etaPlus, updateMax)
                    currUpdateWeight[i][j] = -1 * sign(currentDerivateError[i][j]) * currUpdateValue[i][j]
                elif(currentDerivateError[i][j] * previousDerivateError[i][j] < 0 ):
                    currUpdateValue[i][j] = max(updateValue[i][j] * self.etaMin, updateMin)
                    currUpdateWeight[i][j] = -1 * previousUpdateWeight[i][j]
                    currentDerivateError[i][j] = 0.0
                else:
                    currUpdateValue[i][j] = updateValue[i][j]
                    currUpdateWeight[i][j] = -1 * sign(currentDerivateError[i][j]) * currUpdateValue[i][j]

        return currUpdateValue, currUpdateWeight, currentDerivateError

    def plotError(self):
        numberOfEpochs = len(self.trainCumulativeErrorForEachEpoch)
        epochs = np.arange(numberOfEpochs)
        plt.plot(epochs, self.trainCumulativeErrorForEachEpoch, color='blue', label="Train Error")
        plt.plot(epochs, self.validationCumulativeErrorForEachEpoch, color='green', label="Validation Error")
        plt.ylabel('Errors values')
        plt.xlabel('Numbers of Epochs')
        plt.suptitle('Error trend')
        plt.legend()
        plt.show()

    def vectorialPrediction(self, x):
        self.forwardPropagation(x)
        return self.layers[len(self.layers) - 1].output

    def predict(self, x):
        predict = self.vectorialPrediction(x)
        return np.argmax(predict)

    def remeberThisNet(self, stoppingCriteria, e):
        layersNumber = len(self.layers)
        bestConfiguration = []
        for i in range(layersNumber):
            bestConfiguration.append(self.layers[i].weights)
        stoppingCriteria.setOptWheights(bestConfiguration, e)

    def updateNetThroughBestWeights(self, stoppingCriteria):
        layersNumber = len(self.layers)
        bestConfiguration = stoppingCriteria.getOptWheights()
        for i in range(layersNumber):
            layer = self.layers[i]
            numberOfWeightsForNeuron = len(layer.weights[0])
            neuronsNumber = layer.neuronsNumber

            for r in range(neuronsNumber):
                for c in range(numberOfWeightsForNeuron):
                    layer.weights[r][c] = bestConfiguration[i][r][c]