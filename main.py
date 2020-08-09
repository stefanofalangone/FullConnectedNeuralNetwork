# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import MNIST_Loader as loader
import Function as f
import numpy as np
import Utility as myU
from NeuralNetwork import Net

pathTrainingImages = 'mnist/train-images-idx3-ubyte'
pathTrainingLabels = 'mnist/train-labels-idx1-ubyte'
pathTestImages = 'mnist/t10k-images-idx3-ubyte'
pathTestLabels = 'mnist/t10k-labels-idx1-ubyte'

trainingSet_Images = loader.loadImages(pathTrainingImages) # X independent variable
trainingSet_Labels = loader.loadLabels(pathTrainingLabels) # Y dependent variable
testSet_Images = loader.loadImages(pathTestImages)
testSet_Labels = loader.loadLabels(pathTestLabels)

trainingSet_Images = myU.normalize(trainingSet_Images)
testSet_Images = myU.normalize(testSet_Images)

activationFunctionFirstLayer = f.getSigmoid()
activationFunctionOutputLayer = f.getSoftmax_forCrossEntropy()
errorFunction = f.getCrossEntropy_forSoftmax()

neuronsForLayer = [100,  10]
layersNumber = len(neuronsForLayer)
activationFunctionsLayer = [activationFunctionFirstLayer, activationFunctionOutputLayer]
bias = []

for i in range(layersNumber):
    bias.append(np.ones(neuronsForLayer[i]))

myNet = Net(layersNumber, neuronsForLayer, activationFunctionsLayer, errorFunction, bias)
stoppingCriteriaList = myNet.build(trainingSet_Images, trainingSet_Labels, epochs=200, trainTo=10000, minibatchNumber=10)

for stoppingCriteria in stoppingCriteriaList:
    myNet.evaluateStoppingCriteria(testSet_Images, testSet_Labels, stoppingCriteria, testTo=1000)
"""
# ONLY FOR TESTING PURPOSE.
neuronsForLayerList = [[60,  10], [80,  10], [100,  10], [120,  10], [140,  10], [250, 10]] # <------ change neurons number here
layersNumber = len(neuronsForLayerList[0])
activationFunctionsLayer = [activationFunctionFirstLayer, activationFunctionOutputLayer]

metricsForEachNets = []
for h in range(len(neuronsForLayerList)):
    neuronsForLayer = neuronsForLayerList[h]
    bias = []
    metricsForEachNets.append({})
    metricsForEachNets[h]['netConfiguration'] = ' 784'

    for i in range(layersNumber):
        bias.append(np.ones(neuronsForLayer[i]))
        metricsForEachNets[h]['netConfiguration'] = metricsForEachNets[h]['netConfiguration'] +'x'+ str(neuronsForLayer[i])
    myNet = Net(layersNumber, neuronsForLayer, activationFunctionsLayer, errorFunction, bias)
    stoppingCriteriaList = myNet.build(trainingSet_Images, trainingSet_Labels, epochs=200, trainTo=10000, minibatchNumber=10)

    for stoppingCriteria in stoppingCriteriaList:
        stoppingCriteriaName = stoppingCriteria.__class__.__name__
        metricsForEachNets[h][stoppingCriteriaName] = myNet.evaluateStoppingCriteria(testSet_Images, testSet_Labels, stoppingCriteria, testTo=1000)

beginCenter = '\\begin{center}\n'
beginTabular = '    \\begin{tabular}{||c c c c c c||}\n'
hline = '\\hline\n'
columnsName = '        Net Configuration & AVG Accuracy & Micro F-Score & Stop Epoch & etaPlus & etaMin $\n'
endTabular = '    \\end{tabular}\n'
endCenter = '\\end{center}\n'

for stoppingCriteriaName in ['_GL', '_PQ_disjoint', '_PQ_overlap']:
    File_object = open(stoppingCriteriaName + ".json", "a")
    File_object.write(beginCenter)
    File_object.write(beginTabular)
    File_object.write(hline)
    File_object.write(columnsName)
    File_object.write('\\\\ [0.5ex]\n')
    File_object.write(hline)
    File_object.close()

for metric in metricsForEachNets:
    for stoppingCriteriaName in ['_GL', '_PQ_disjoint', '_PQ_overlap']:
        resultList = metric.get(stoppingCriteriaName)
        File_object = open(stoppingCriteriaName+".json", "a")

        File_object.write(hline)
        row = '        '
        row = row + metric.get('netConfiguration') + ' & '

        for i in range(len(resultList)):
            if(i < len(resultList)-1 and i > 0):
                row = row + str(np.round(resultList[i], decimals=3)) + ' & '
            elif (i == len(resultList)-1):
                row = row + str(resultList[i]) + '  '
        row = row + '1.6 & 0.6 \\\\'
        row = row + '[1ex]\n'
        File_object.write(row)
        File_object.close()

for stoppingCriteriaName in ['_GL', '_PQ_disjoint', '_PQ_overlap']:
    File_object = open(stoppingCriteriaName + ".json", "a")
    File_object.write(hline)
    File_object.write(endTabular)
    File_object.write(endCenter)
    File_object.close()
"""