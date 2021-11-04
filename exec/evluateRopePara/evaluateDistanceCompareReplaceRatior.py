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
        print(len(mergedTrajectories),filesNames)
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
 

    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.0,0.5]#[0.0, 1.0]
    manipulatedVariables['frictionloss'] =[0.0,1.0]# [0.0, 0.2, 0.4] 
    manipulatedVariables['ropeLength'] = [0.06]# [0.06,0.09]#[0.0, 1.0]
    manipulatedVariables['killZone'] = [4.5]#[0.0, 1.0]
    manipulatedVariables['killZoneofDistractor'] = [0.0]#[0.0, 1.0]
    # manipulatedVariables['wolfMass'] =[2.0]# [0.0, 0.2, 0.4]
    # manipulatedVariables['distractorNoise'] = [0.0,3.0,6.0,9.0]#[0.0, 1.0]
    # manipulatedVariables[''] =[1.0]# [0.0, 0.2, 0.4]
    # manipulatedVariables['']=[0.0]#[0.0, 2.0]
    # manipulatedVariables['offset'] = [int(-2),int(-1),int(0),int(1),int(2)]
    manipulatedVariables['offset'] = [0.0]
    
    manipulatedVariables['hideId'] = [6]

    allAgentNames =  ['wolf','sheep','master','disractor1','disractor2']
    agentIdName = allAgentNames.copy()
    # del(agentIdName[manipulatedVariables['hideId'][0]])
    # damping = 0.5
    # frictionloss = 1.0
    masterForce = 2.0
    sheepForce = 5.0
    # wolfMass = 1.0

    killZone = 2.0
    ropePunishWeight = 0.3
    # ropeLength = 0.12
    masterMass = 2.0
    distractorNoise = 0.0

    folderNameList = ['masterForceAndMass','masterAndWolfMass']
    folderName = folderNameList[0]
    evalNum=50
    evaluateEpisode=60000



    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)




    dataFolder = os.path.join(dirName, '..','..', 'data')
    # trajectoryDirectory= os.path.join(dataFolder,'trajectory','noiseOffsetMasterForSelect6.8')
    # trajectoryDirectory = os.path.join(dataFolder, 'Exptrajectory', modelSaveName,'noiseOffsetMasterForSelectOct11')
    # trajectoryDirectory = os.path.join(dataFolder, 'trajectory', 'noiseOffsetMasterForSelectOct19len0.12')
    # trajectoryDirectory = os.path.join(dataFolder, 'trajectory', 'noiseOffsetMasterForSelectOctVarRopeLength')
    # trajectoryDirectory = os.path.join(dataFolder, 'trajectory', 'noiseOffsetMasterForSelectOct20VarNoise')
    # trajectoryDirectory = os.path.join(dataFolder, 'trajectory', 'noiseOffsetMasterForSelectOct21')
    trajectoryDirectory = os.path.join(dataFolder, 'trajectory', 'Oct26',folderName)
    trajectoryExtension = '.pickle'


    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'masterForce':masterForce,'distractorNoise':distractorNoise,'ropePunishWeight':ropePunishWeight,'killZone':killZone,'masterMass':masterMass,'ropeLength':ropeLength}
    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'distractorNoise':distractorNoise,'ropePunishWeight':ropePunishWeight,'killZone':killZone}
    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'ropePunishWeight':ropePunishWeight,'masterMass':masterMass,'distractorNoise':distractorNoise}
    trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'masterForce':masterForce,'ropePunishWeight':ropePunishWeight,'masterMass':masterMass,'distractorNoise':distractorNoise}
    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'distractorNoise':distractorNoise,'ropePunishWeight':ropePunishWeight,'killZone':killZone,'masterMass':masterMass,'ropeLength':ropeLength}
    # trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'damping':damping,'frictionloss':frictionloss,'masterForce':masterForce,'distractorNoise':distractorNoise,'ropePunishWeight':ropePunishWeight,'killZone':killZone,'masterMass':masterMass,'ropeLength':ropeLength,'wolfMass':wolfMass,'sheepForce':sheepForce}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    def calChasingSubPair(tragetAgentsId):  
        measurementFunction = lambda trajectory: calculateChasingSubtlety(trajectory,tragetAgentsId)
        # measurementFunction = lambda trajectory: calculateDistance(trajectory,tragetAgentsId)
        computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
        statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

        statisticsDf_ = statisticsDf.reset_index()

        statisticsDf_['{}->{}'.format(agentIdName[tragetAgentsId[0]],agentIdName[tragetAgentsId[1]])] =statisticsDf_.loc[:,'mean']

        print(statisticsDf_)
        return statisticsDf_

    evlueatePairs = [[0,1],[2,0],[2,3]]#for hide distractor
    
    
    data = [calChasingSubPair(Ids) for Ids in evlueatePairs]
    # df = data[0].join(data[1],on=['ropeLength','distractorNoise','offset','hideId'])
    tomergeKeys = list(manipulatedVariables.keys())
    df=pd.merge(data[0],data[1],on=tomergeKeys)
    df2 = pd.merge(df,data[2],on=tomergeKeys)
    print(df)
    csvfilePath = os.path.join(trajectoryDirectory,'{}SubtletyMasterForce.csv'.format(folderName))
    # csvfilePath = os.path.join(trajectoryDirectory,'{}Distance.csv'.format(folderName))
    df2.dropna(axis=0)
    print(df2)
    df2.to_csv(csvfilePath)
    lableList = ['{}->{}'.format(agentIdName[a],agentIdName[b]) for a,b in evlueatePairs]

    # tragetAgentsId1 = [0,1]
    # dfwolf_sheep =  calChasingSubPair(tragetAgentsId1)  
    # tragetAgentsId2 = [0,2]
    # dfwolf_distra1 = calChasingSubPair(tragetAgentsId2) 
    # tragetAgentsId3 = [0,3]
    # dfwolf_distra2 = calChasingSubPair(tragetAgentsId3) 

    # tragetAgentsId4 = [1,2]
    # dfsheep_distra1 = calChasingSubPair(tragetAgentsId4) 
    # tragetAgentsId5 = [1,3]
    # dfsheep_distra2 = calChasingSubPair(tragetAgentsId5) 

    # lableList = ['{}->{}'.format(agentIdName[a],agentIdName[b]) for a,b in [tragetAgentsId1,tragetAgentsId2,tragetAgentsId3,tragetAgentsId4,tragetAgentsId5]]

    # measurementFunction2 = lambda trajectory: calculateChasingSubtlety(trajectory,tragetAgentsId2)
    # computeStatistics2 = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction2)
    # statisticsDf2 = toSplitFrame.groupby(levelNames).apply(computeStatistics2)
    # statisticsDf2_ = statisticsDf2.reset_index()
    # dfwolf_distra1 = statisticsDf2_.groupby('offset').mean()

    # tragetAgentsId3 = [1,2]
    # measurementFunction3 = lambda trajectory: calculateChasingSubtlety(trajectory,tragetAgentsId3)
    # computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction3)
    # statisticsDf3 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    # statisticsDf3_ = statisticsDf3.reset_index()
    # dfsheep_distra1 = statisticsDf3_.groupby('offset').mean()

    # print(dfwolf_sheep,dfwolf_distra1,dfsheep_distra1,dfwolf_distra2,dfsheep_distra2)


    from matplotlib import pyplot as plt
    fig = plt.figure()
    axForDraw = fig.add_subplot(1,1,1)
    # dfwolf_sheep.plot(ax=axForDraw, label='wolf->sheep', y='mean',marker='o',color='green', logx=False)
    # dfwolf_distra1.plot(ax=axForDraw, label='wolf->distra1', y='mean',marker='o',color='red', logx=False)
    # dfsheep_distra1.plot(ax=axForDraw, label='sheep->distra1', y='mean',marker='o',color='blue', logx=False)
    # dfwolf_distra2.plot(ax=axForDraw, label='wolf->distra2', y='mean',marker='o',color='brown', logx=False)
    # dfsheep_distra2.plot(ax=axForDraw, label='sheep->distra2', y='mean',marker='o',color='orange', logx=False)
    # data = [dfwolf_sheep,dfwolf_distra1,dfwolf_distra2,dfsheep_distra1,dfsheep_distra2]
    # print(dfwolf_sheep['offset'=0.0]['mean'])
    # print(dfwolf_sheep.values)
    toDrawData = [df.values[0][1]+0.5 for df in data]
    toDrawData2 = [df.values[0][1]+1.5 for df in data]

    plt.bar(range(len(toDrawData)),toDrawData,tick_label=lableList,color='red')
    plt.bar(range(len(toDrawData)),toDrawData2,bottom=toDrawData,tick_label=lableList,color='green')
    
    axForDraw.set_ylim(0, 2)
    # axForDraw.set_ylim(30, 120)
    # plt.suptitle('sheepCrossLeashPerTraj\n50trajs\nropePunishWeight={}killZone={}ropeLength={}'.format(ropePunishWeight,killZone,ropeLength))
    # plt.suptitle('AverageChasingSubtlety\nmasterMass={}killZone={}ropeLength={}'.format(masterMass,killZone,ropeLength))
    # plt.suptitle('AverageChasingSubtletyHideAgent={}\nmasterForce={}sheepForce={}wolfMass={}'.format(allAgentNames[manipulatedVariables['hideId'][0]],masterForce,sheepForce,wolfMass))
    # plt.suptitle('AverageDistanceyHideAgent={}\nmasterForce={}sheepForce={}wolfMass={}'.format(allAgentNames[manipulatedVariables['hideId'][0]],masterForce,sheepForce,wolfMass))
    # plt.suptitle('AverageDistanceyHideAgent={}\ndamping={}frictionloss={}masterForce={}'.format(allAgentNames[manipulatedVariables['hideId'][0]],damping,frictionloss,masterForce))

    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
