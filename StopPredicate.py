import numpy as np

def getGl(alpha = 1.0):
    return _GL(alpha)

def getPq_disjoint(strip=5, alpha=0.5):
    return _PQ_disjoint(strip, alpha)

def getPq_overlap(strip=5, alpha=0.5):
    return _PQ_overlap(strip, alpha)

def getNoStopCriteria():
    return NoStop()

def getNaiveFirstIncreaseCriteria():
    return FisrtIncrease()

class StopPredicate:
    def __init__(self):
        self.optWeights = []
        self.epochOfOptWeights = None
        self.stopEpoch = None

    #(NEW) Forse da togliere, che PQ non lo usa
    def shouldEarlyStop(self, x, epoch):
        pass

    def getOptWheights(self):
        return self.optWeights

    def setOptWheights(self, weightsToSet, e):
        self.optWeights=weightsToSet
        self.epochOfOptWeights = e

    def getStopEpoch(self):
        return self.stopEpoch

    def getBestEpoch(self):
        return self.epochOfOptWeights

    def getE_opt(self):
        pass

class NoStop(StopPredicate):
    def __init__(self):
        super(NoStop, self).__init__()
        self.E_opt = np.inf

    def shouldEarlyStop(self, validationError, epoch):
        if(validationError<self.E_opt):
            self.E_opt = validationError
        return False

    def getE_opt(self):
        return self.E_opt

class FisrtIncrease(StopPredicate):
    def __init__(self):
        super(FisrtIncrease, self).__init__()
        self.lastError = np.inf
        
    def shouldEarlyStop(self, validationError, epoch):
        if validationError > self.lastError:
            self.lastError=validationError
            return False
        else:
            self.stopEpoch=epoch
            return True

    def getE_opt(self):
        return self.lastError

class _GL(StopPredicate):
    def __init__(self, alpha=1.0):
        super(_GL, self).__init__()
        self.optError = np.inf
        self.alpha = alpha

    def shouldEarlyStop(self, validationError, epoch):
        gl = self.calculateGl(validationError)
        print("GL value: ", gl)
        if (gl > self.alpha):
            self.stopEpoch=epoch
            return True
        else:
            return False

    def calculateGl(self, validationError):
        self.updateE_opt(validationError)
        return 100 * ((validationError/self.optError) - 1)

    def updateE_opt(self, validationError):
        if validationError<self.optError:
            self.optError = validationError

    def getE_opt(self):
        return self.optError

class _PQ_disjoint(StopPredicate):

    def __init__(self, stripSize=5, alpha=0.5):
        super(_PQ_disjoint, self).__init__()
        self.stripSize = stripSize
        self.alpha = alpha
        self.curr = 0
        self.trainingErrors = np.zeros(stripSize)
        self.gl = getGl()

    def shouldEarlyStop(self, trainError, validationError, epoch):
        self.addError(trainError, validationError)
        if self.curr != self.stripSize:
            return False
        pk = self.calculatePk(self.trainingErrors)
        glValue = self.gl.calculateGl(validationError)

        self.curr = 0
        print("PQ_disjoint value: ", glValue/pk)
        if (glValue/pk > self.alpha):
            self.stopEpoch=epoch
            return True
        else:
            return False

    def addError(self, trainError, validationError):
        self.trainingErrors[self.curr] = trainError
        self.curr = self.curr + 1
        self.gl.updateE_opt(validationError)

    def calculatePk(self, trainError):
        minError = np.min(trainError)
        cumulativeError = np.sum(trainError)
        return 1000 * ((cumulativeError/(self.stripSize * minError)) - 1)

    def getStrip(self):
        return self.stripSize

    def getE_opt(self):
        return self.gl.optError

class _PQ_overlap(StopPredicate):

    def __init__(self, stripSize=5, alpha=0.5):
        super(_PQ_overlap, self).__init__()
        self.stripSize = stripSize
        self.alpha = alpha
        self.curr = 0
        self.isFull = False
        self.trainingErrors = np.zeros(stripSize)
        self.gl = getGl()

    def shouldEarlyStop(self, trainError, validationError, epoch):
        self.addError(trainError, validationError)
        if not(self.isFull):
            return False
        pk = self.calculatePk(self.trainingErrors)
        glValue = self.gl.calculateGl(validationError)
        print("PQ_overlap value: ", glValue/pk)
        if (glValue/pk > self.alpha):
            self.stopEpoch=epoch
            return True
        else:
            return False

    def addError(self, trainingError, validationError):
        self.trainingErrors[self.curr] = trainingError
        self.curr = (self.curr + 1) % self.stripSize
        self.gl.updateE_opt(validationError)
        self.isFull = self.isFull or (self.curr==0)

    def calculatePk(self, trainingErrors):
        minTError = np.min(trainingErrors)
        sumTError = np.sum(trainingErrors)
        return 1000 * ((sumTError/(self.stripSize * minTError)) - 1)

    def getStrip(self):
        return self.stripSize

    def getE_opt(self):
        return self.gl.optError