import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# import xmltodict
# import mujoco_py as mujoco
import pandas as pd
import itertools as it
from collections import OrderedDict
import numpy as np
import glob
# from env.multiAgentMujocoEnv import RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
#     getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed

# from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath,loadFromPickle
# from src.functionTools.trajectory import SampleExpTrajectory
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
# from src.visualize.visualizeMultiAgent import Render



wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
distractorColor = np.array([0.35, 0.85, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])




class ReshapeAction:
    def __init__(self,sensitivity):
        self.actionDim = 2
        self.sensitivity = sensitivity

    def __call__(self, action): # action: tuple of dim (5,1)
        # print(action)
        actionX = action[1] - action[2]
        actionY = action[3] - action[4]
        actionReshaped = np.array([actionX, actionY]) * self.sensitivity
        # print(actionReshaped,'2d')
        return actionReshaped

def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters

class LoadTrajectories:
    def __init__(self, getSavePath, loadFromPickle, fuzzySearchParameterNames=[]):
        self.getSavePath = getSavePath
        self.loadFromPickle = loadFromPickle
        self.fuzzySearchParameterNames = fuzzySearchParameterNames

    def __call__(self, parameters, parametersWithSpecificValues={}):
        parametersWithFuzzy = dict(list(parameters.items()) + [(parameterName, '*') for parameterName in self.fuzzySearchParameterNames])
        productedSpecificValues = it.product(*[[(key, value) for value in values] for key, values in parametersWithSpecificValues.items()])
        parametersFinal = np.array([dict(list(parametersWithFuzzy.items()) + list(specificValueParameter)) for specificValueParameter in productedSpecificValues])
        genericSavePath = [self.getSavePath(parameters) for parameters in parametersFinal]
        if len(genericSavePath) != 0:
            filesNames = np.concatenate([glob.glob(savePath) for savePath in genericSavePath])
        else:
            filesNames = []
        mergedTrajectories = []
        for fileName in filesNames:
            oneFileTrajectories = self.loadFromPickle(fileName)
            mergedTrajectories.extend(oneFileTrajectories)
        return mergedTrajectories
def calculateChasingSubtlety(traj):
    # print(len(traj))

    reshapeActon = ReshapeAction(5)
    def calculateIncludedAngle(vector1,vector2):
        # print(vector1,vector2)
        v1=complex(vector1[0],vector1[1])
        v2=complex(vector2[0],vector2[1])

        return np.abs(np.angle(v1/v2))*180/np.pi
    # for traj in trajs:
    # wolfSheepAngle=np.mean([calculateIncludedAngle(np.array(state[0][0][2:4]),np.array(state[0][1][0:2])-np.array(state[0][0][0:2])) for state in  traj    ])
    # wolfSheepAngle=np.mean([calculateIncludedAngle(np.array(traj[index][1])-np.array(traj[index][0]),np.array(traj[index+1][0])-np.array(traj[index][0])) for index in  range(0,len(traj)-1)    ])
    # wolfSheepAngleList.append(wolfSheepAngle)
    # averageAngle = np.mean(wolfSheepAngleList)

    # wolfMasterAngle=np.mean([calculateIncludedAngle(np.array(state[0][2][2:4]),np.array(state[0][0][0:2])-np.array(state[0][2][0:2])) for state in  traj    ])
    wolfMasterAngle=np.mean([calculateIncludedAngle(np.array(traj[index][0])-np.array(traj[index][2]),np.array(traj[index+1][2])-np.array(traj[index][2])) for index in  range(0,len(traj)-1)])
    # print(wolfMasterAngle)

    return wolfMasterAngle
def calculateWolfSheepChasingSubtlety(traj):
    # print(len(traj))

    reshapeActon = ReshapeAction(5)
    def calculateIncludedAngle(vector1,vector2):
        # print(vector1,vector2)
        v1=complex(vector1[0],vector1[1])
        v2=complex(vector2[0],vector2[1])

        return np.abs(np.angle(v1/v2))*180/np.pi
    # for traj in trajs:
    # wolfSheepAngle=np.mean([calculateIncludedAngle(np.array(state[0][0][2:4]),np.array(state[0][1][0:2])-np.array(state[0][0][0:2])) for state in  traj    ])
    wolfSheepAngle=np.mean([calculateIncludedAngle(np.array(traj[index][1])-np.array(traj[index][0]),np.array(traj[index+1][0])-np.array(traj[index][0])) for index in  range(0,len(traj)-1)    ])
    # wolfSheepAngleList.append(wolfSheepAngle)
    # averageAngle = np.mean(wolfSheepAngleList)

    # wolfMasterAngle=np.mean([calculateIncludedAngle(np.array(state[0][2][2:4]),np.array(state[0][0][0:2])-np.array(state[0][2][0:2])) for state in  traj    ])
    # wolfMasterAngle=np.mean([calculateIncludedAngle(np.array(traj[index][0])-np.array(traj[index][2]),np.array(traj[index+1][2])-np.array(traj[index][2])) for index in  range(0,len(traj)-1)])
    # print(wolfMasterAngle)

    return wolfSheepAngle

def calculateDistance(traj):
    # print(len(traj))
    tragetAgentId = [0,1]
    def calculateDistance(pointA,pointB):
        
        return np.linalg.norm(pointA - pointB ,ord =2)

    wolfMasterAngle=np.mean([calculateDistance(np.array(traj[index][tragetAgentId[0]]),np.array(traj[index][tragetAgentId[1]])) for index in  range(0,len(traj))])
    # print(wolfMasterAngle)

    return wolfMasterAngle

def calculateCrossLeash(traj):
    # print(len(traj))

    def calculateCross(A,B,C,D):
        lineAC = C - A
        lineAD = D - A
        lineBC = C - B 
        lineBD = D - B
        if (min(A[0] ,B[0])<= max(C[0],D[0])) and (min(C[0] ,D[0])<= max(A[0],B[0])) and (min(A[1] ,B[1])<= max(C[1],D[1])) and (min(C[1] ,D[1])<= max(A[1],B[1])):
            def vectorProduct(vector1,vector2):
                return vector1[0] * vector2[1] - vector2[0] * vector1[1]
            return (vectorProduct(lineAC,lineAD)*vectorProduct(lineBC,lineBD)<=0) and (vectorProduct(lineAC,lineBC)*vectorProduct(lineAD,lineBD)<=0)
        else:
            return False

    crossRatio=np.mean([calculateCross(np.array(traj[index][0]),np.array(traj[index][2]),np.array(traj[index][1]),np.array(traj[index+1][1])) for index in  range(0,len(traj)-1)])
    # print(wolfMasterAngle)

    return crossRatio

class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = np.array([self.measurementFunction(trajectory) for trajectory in allTrajectories])
        # print(oneConditionDf)
        measurementMean = np.mean(allMeasurements, axis = 0)
        measurementStd = np.std(allMeasurements, axis = 0)
        return pd.Series({'mean': measurementMean, 'std': measurementStd})

def main():
    # manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.0,1.0]#[0.0, 1.0]
    # manipulatedVariables['frictionloss'] =[0.0,0.2]# [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce']=[0.0,1.0]#[0.0, 2.0]


    manipulatedVariables = OrderedDict()
    # manipulatedVariables[''] = [0.5]#[0.0, 1.0]
    # manipulatedVariables[''] =[1.0]# [0.0, 0.2, 0.4]
    # manipulatedVariables['']=[0.0]#[0.0, 2.0]
    # manipulatedVariables['offset'] = [int(-2),int(-1),int(0),int(1),int(2)]
    manipulatedVariables['offset'] = [0]
    manipulatedVariables['hideId'] = [3,4]

    damping = 0.5
    frictionloss = 1.0
    masterForce = 1.0

    distractorNoise = 3.0

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    evalNum=50
    evaluateEpisode=120000

    # for condition in conditions:
    # #     print(condition)
    #     # generateSingleCondition(condition)
    #     try:
    #         generateSingleCondition(condition)
    #     except:
    #         continue


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)




    dataFolder = os.path.join(dirName, '..','..', 'data')
    trajectoryDirectory= os.path.join(dataFolder,'trajectory','noiseOffsetMasterForSelect6.8')
    trajectoryExtension = '.pickle'

    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
    trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'distractorNoise':distractorNoise}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: calculateChasingSubtlety(trajectory)

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)


    # print(statisticsDf)

    # manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0,1.0]#[0.0, 1.0]
    # manipulatedVariables['frictionloss'] =[0,0.2]# [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce']=[0,1.0]#[0.0, 2.0]


    # statisticsDf2 = toSplitFrame2.groupby(levelNames2).apply(computeStatistics)

    # print(statisticsDf)
    # print(statisticsDf2)

    statisticsDf3 = statisticsDf.reset_index()

    measurementFunction2 = lambda trajectory: calculateWolfSheepChasingSubtlety(trajectory)
    computeStatistics2 = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction2)
    statisticsDf2 = toSplitFrame.groupby(levelNames).apply(computeStatistics2)
    baseLine = statisticsDf2.mean()
    print(statisticsDf2)
    print(baseLine)
    # statisticsDf4 = statisticsDf2.reset_index()
    # df = statisticsDf3.append(statisticsDf4)
    # pdAll = pd.merge(statisticsDf3,statisticsDf4)
    df = statisticsDf3.groupby('offset').mean()
    # df2 = df.groupby('hideId').mean()
    print(df)



    from matplotlib import pyplot as plt
    fig = plt.figure()
    axForDraw = fig.add_subplot(1,1,1)
    df.plot(ax=axForDraw, label='master-wolf', y='mean',marker='o', logx=False)
    plt.hlines(baseLine, -0.5,1.0, colors = "r", linestyles = "dashed")
    axForDraw.set_ylim(40, 140)
    plt.suptitle('chasing subtlety(baseline = wolf-sheep\ndamping={}frictionloss={}masterForce={}'.format(damping,frictionloss,masterForce))

    plt.legend(loc='best')
    plt.show()
    # meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='displayTime={}'.format(displayTime))
    # plt.bar(displayTime, meanSub, label='displayTime={}'.format(displayTime), align='center')
    # plt.hlines(1/6, -0.5,1.5, colors = "r", linestyles = "dashed")
    # fig.set_xlim(-0, 1)
#     fig = plt.figure()
#     numRows = len(manipulatedVariables['damping'])
#     numColumns = len(manipulatedVariables['frictionloss'])

#     def drawPerformanceLine(dataDF,axForDraw):
#         dataDF.plot(ax=axForDraw,label='masterForce',y='mean',yerr='std',marker='o',logx=False)

#     plotCounter = 1
#     for damping, grp in statisticsDf.groupby('damping'):
#         grp.index = grp.index.droplevel('damping')

#         for frictionloss, group in grp.groupby('frictionloss'):
#             group.index = group.index.droplevel('frictionloss')

#             axForDraw = fig.add_subplot(numRows,numColumns,plotCounter)
#             if plotCounter % numColumns == 1:
#                 axForDraw.set_ylabel('damping: {}'.format(damping))
#             if plotCounter <= numColumns:
#                 axForDraw.set_title('frictionloss: {}'.format(frictionloss))
#             axForDraw.set_ylim(0, 180)
# #
#             # plt.ylabel('chasing subtlety')
#             drawPerformanceLine(group, axForDraw)
#             # trainStepLevels = statisticsDf.index.get_level_values('trainSteps').values
#             # axForDraw.plot(trainStepLevels, [1.18]*len(trainStepLevels), label='mctsTrainData')
#             plotCounter += 1
#     plt.suptitle('chasing subtlety')

#     plt.legend(loc='best')
#     plt.show()
if __name__ == '__main__':
    main()
