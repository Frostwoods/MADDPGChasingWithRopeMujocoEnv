import os
import pickle
import itertools as it
import numpy as np
import glob
class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)
        return path


def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object

def restoreVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.restore(model, path)
    print("Model restored from {}".format(path))
    return model

class LoadTrajectories:
    def __init__(self, getSavePath, loadFromPickle, fuzzySearchParameterNames=[]):
        self.getSavePath = getSavePath
        self.loadFromPickle = loadFromPickle
        self.fuzzySearchParameterNames = fuzzySearchParameterNames

    def __call__(self, parameters, parametersWithSpecificValues={}):
        parametersWithFuzzy = dict(
            list(parameters.items()) + [(parameterName, '*') for parameterName in self.fuzzySearchParameterNames])
        productedSpecificValues = it.product(
            *[[(key, value) for value in values] for key, values in parametersWithSpecificValues.items()])
        parametersFinal = np.array(
            [dict(list(parametersWithFuzzy.items()) + list(specificValueParameter)) for specificValueParameter in
             productedSpecificValues])
        genericSavePath = [self.getSavePath(parameters) for parameters in parametersFinal]
        if len(genericSavePath) != 0:
            filesNames = np.concatenate([sorted(glob.glob(savePath)) for savePath in genericSavePath])
        else:
            filesNames = []
        mergedTrajectories = []
        for fileName in filesNames:
            print(fileName)
            oneFileTrajectories = self.loadFromPickle(fileName)
            mergedTrajectories.extend(oneFileTrajectories)
        return mergedTrajectories

def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters
