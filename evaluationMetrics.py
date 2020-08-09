import numpy as np

class EvaluationMetrics:
    accuracy = []
    precision = []
    recall = []
    fScore = []

    def __init__(self, truePositives, falsePositives, falseNegatives, testSize, beta):
        trueNegatives = testSize-(truePositives+falsePositives+falseNegatives)
        self.guessRate = np.sum(truePositives)/testSize
        self.accuracy = (truePositives+trueNegatives)/testSize
        self.precision = truePositives/(truePositives+falsePositives)
        self.recall = truePositives/(truePositives+falseNegatives)
        self.fScore = ( beta*beta + 1 )* self.precision*self.recall / (beta*beta*self.precision + self.recall)

        classes = np.size(truePositives)
        self.avgAccuracy = np.sum(self.accuracy)/classes
        self.macroPrecision = np.sum(self.precision)/classes
        self.macroRecall = np.sum(self.recall)/classes
        self.microPrecision = np.sum(truePositives)/np.sum(truePositives+falsePositives)
        self.microRecall = np.sum(truePositives)/np.sum(truePositives+falseNegatives)
        self.microFScore = ( beta*beta + 1 )* self.microPrecision*self.microRecall / (beta*beta*self.microPrecision + self.microRecall)
        self.macroFScore = ( beta*beta + 1 )* self.macroPrecision*self.macroRecall / (beta*beta*self.macroPrecision + self.macroRecall)

    def getGuessRate(self):
        return self.guessRate

    def getAccuracy(self, of=None):
        if of is None:
            return self.accuracy
        else:
            try:
                return self.accuracy[of]
            except:
                return self.accuracy

    def getPrecision(self, of=None):
        if of is None:
            return self.precision
        else:
            try:
                return self.precision[of]
            except:
                return self.precision

    def getRecall(self, of=None):
        if of is None:
            return self.recall
        else:
            try:
                return self.recall[of]
            except:
                return self.recall

    def getFScore(self, of=None):
        if of is None:
            return self.fScore
        else:
            try:
                return self.fScore[of]
            except:
                return self.fScore

    def getAVGAccuracy(self):
        return self.avgAccuracy

    def getMacroPrecision(self):
        return self.macroPrecision

    def getMacroRecall(self):
        return self.macroRecall

    def getMacroFScore(self):
        return self.macroFScore

    def getMicroPrecision(self):
        return self.microPrecision

    def getMicroRecall(self):
        return self.microRecall

    def getMicroFScore(self):
        return self.microFScore

def getEvaluationMetrics(truePositives, falsePositives, falseNegatives, testSize, beta=2.0):
    return EvaluationMetrics(truePositives, falsePositives, falseNegatives, testSize, beta)