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


    return wolfSheepAngle
def calculateChasingSubtlety(traj,tragetAgentsId):

    def calculateIncludedAngle(vector1,vector2):
        # print(vector1,vector2)
        v1=complex(vector1[0],vector1[1])
        v2=complex(vector2[0],vector2[1])

        return np.abs(np.angle(v1/v2))*180/np.pi

    wolfMasterAngle=np.mean([calculateIncludedAngle(np.array(traj[index][tragetAgentsId[1]])-np.array(traj[index][tragetAgentsId[0]]),np.array(traj[index+1][tragetAgentsId[0]])-np.array(traj[index][tragetAgentsId[0]])) for index in  range(0,len(traj)-1)])
    # print(wolfMasterAngle)

    return wolfMasterAngle
def calculateDistance(traj,tragetAgentsId):
    # print(len(traj))
    
    def calculateDistance(pointA,pointB):
        
        return np.linalg.norm(pointA - pointB ,ord =2)

    wolfMasterAngle=np.mean([calculateDistance(np.array(traj[index][tragetAgentsId[0]]),np.array(traj[index][tragetAgentsId[1]])) for index in  range(0,len(traj))])
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

    crossNum = np.sum([calculateCross(np.array(traj[index][0]),np.array(traj[index][2]),np.array(traj[index][1]),np.array(traj[index+1][1])) for index in  range(0,len(traj)-1)])
    # print(wolfMasterAngle)

    return crossNum 

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
    manipulatedVariables['offset'] = [0.0]
    
    manipulatedVariables['hideId'] = [2]

    damping = 0.5
    frictionloss = 1.0
    masterForce = 1.0
    killZone = 4.0
    ropePunishWeight = 0.3
    ropeLength = 0.06
    masterMass = 1.0
    distractorNoise = 0.0

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
    # trajectoryDirectory= os.path.join(dataFolder,'trajectory','noiseOffsetMasterForSelect6.8')
    modelSaveName = 'expTrajMADDPGMujocoEnvOct'
    # trajectoryDirectory = os.path.join(dataFolder, 'Exptrajectory', modelSaveName,'noiseOffsetMasterForSelectOct11')
    trajectoryDirectory = os.path.join(dataFolder, 'trajectory', 'noiseOffsetMasterForSelectOct14')
    trajectoryExtension = '.pickle'

    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'distractorNoise':distractorNoise}
    trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'distractorNoise':distractorNoise,'ropePunishWeight':ropePunishWeight,'killZone':killZone,'masterMass':masterMass,'ropeLength':ropeLength}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    tragetAgentsId = [0,1]
    measurementFunction = lambda trajectory: calculateChasingSubtlety(trajectory,tragetAgentsId)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    statisticsDf_ = statisticsDf.reset_index()
    dfwolf_sheep = statisticsDf_.groupby('offset').mean()

    tragetAgentsId2 = [2,0]
    measurementFunction2 = lambda trajectory: calculateChasingSubtlety(trajectory,tragetAgentsId2)
    computeStatistics2 = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction2)
    statisticsDf2 = toSplitFrame.groupby(levelNames).apply(computeStatistics2)
    statisticsDf2_ = statisticsDf2.reset_index()
    dfwolf_master = statisticsDf2_.groupby('offset').mean()



    tragetAgentsId3 = [2,1]
    measurementFunction3 = lambda trajectory: calculateChasingSubtlety(trajectory,tragetAgentsId3)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction3)
    statisticsDf3 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    statisticsDf3_ = statisticsDf3.reset_index()
    dfsheep_master = statisticsDf3_.groupby('offset').mean()

    print(dfwolf_sheep,dfwolf_master,dfsheep_master)


    from matplotlib import pyplot as plt
    fig = plt.figure()
    axForDraw = fig.add_subplot(1,1,1)
    dfwolf_sheep.plot(ax=axForDraw, label='wolf->sheep', y='mean',marker='o',color='green', logx=False)
    dfwolf_master.plot(ax=axForDraw, label='master->wolf', y='mean',marker='o',color='red', logx=False)
    dfsheep_master.plot(ax=axForDraw, label='master->sheep', y='mean',marker='o',color='blue', logx=False)
    axForDraw.set_ylim(30, 120)
    # plt.suptitle('sheepCrossLeashPerTraj\n50trajs\nropePunishWeight={}killZone={}ropeLength={}'.format(ropePunishWeight,killZone,ropeLength))
    plt.suptitle('AverageChasingSubtlety\nmasterMass={}killZone={}ropeLength={}'.format(masterMass,killZone,ropeLength))

    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
