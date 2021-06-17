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




# def generateSingleCondition(condition):
#     debug = 0
#     if debug:


#         damping=2.0
#         frictionloss=0.0
#         masterForce=1.0

#         numWolves = 1
#         numSheeps = 1
#         numMasters = 1
#         numDistractor = 1
#         maxTimeStep = 25

#         maxEpisode = 60000
#         saveTraj=True
#         saveImage=True
#         visualizeMujoco=False
#         visualizeTraj = True
#         makeVideo=True
#     else:

#         # print(sys.argv)
#         # condition = json.loads(sys.argv[1])
#         damping = float(condition['damping'])
#         frictionloss = float(condition['frictionloss'])
#         masterForce = float(condition['masterForce'])

#         maxEpisode = 120000
#         evaluateEpisode = 120000
#         numWolves = 1
#         numSheeps = 1
#         numMasters = 1
#         numDistractor = 1
#         maxTimeStep = 25

#         saveTraj=True
#         saveImage=True
#         visualizeMujoco=False
#         visualizeTraj = False
#         makeVideo=False

#     evalNum = 100
#     maxRunningStepsToSample = 100
#     modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
#     print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))


#     numAgent = numWolves + numSheeps + numMasters +  numDistractor
#     wolvesID = [0]
#     sheepsID = [1]
#     masterID = [2]
#     distractorID = [3]

#     wolfSize = 0.05
#     sheepSize = 0.05
#     masterSize = 0.05
#     distractorSize = 0.05
#     entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters + [distractorSize] * numDistractor


#     entitiesMovableList = [True] * numAgent + [False] * numMasters

#     killZone = 0.01
#     isCollision = IsCollision(getPosFromAgentState, killZone)
#     punishForOutOfBound = PunishForOutOfBound()
#     rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)
#     rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
#     rewardDistractor = RewardSheep(wolvesID+sheepsID+masterID, distractorID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)
#     rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
#     rewardFunc = lambda state, action, nextState: \
#         list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))\
#         + list(rewardMaster(state, action, nextState) )+ list(rewardDistractor(state, action, nextState))

#     physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leasedAddDistractorForExp.xml')
#     with open(physicsDynamicsPath) as f:
#         xml_string = f.read()

#     makePropertyList=MakePropertyList(transferNumberListToStr)

#     geomIds=[1,2,3,4]
#     keyNameList=[0,1]
#     valueList=[[damping,damping]]*len(geomIds)
#     dampngParameter=makePropertyList(geomIds,keyNameList,valueList)

#     changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

#     geomIds=[1,2,3,4]
#     keyNameList=[0,1]
#     valueList=[[frictionloss,frictionloss]]*len(geomIds)
#     frictionlossParameter=makePropertyList(geomIds,keyNameList,valueList)
#     changeJointFrictionlossProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@frictionloss')



#     envXmlDict = xmltodict.parse(xml_string.strip())
#     envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
#     changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
#     for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
#         envXmlDict=changeXmlProperty(envXmlDict,propertyDict)

#     envXml=xmltodict.unparse(envXmlDict)
#     physicsModel = mujoco.load_model_from_xml(envXml)
#     physicsSimulation = mujoco.MjSim(physicsModel)

#     numKnots = 9
#     numAxis = (numKnots + numAgent) * 2
#     qPosInit = (0, ) * numAxis
#     qVelInit = (0, ) * numAxis
#     qPosInitNoise = 0.4
#     qVelInitNoise = 0
#     tiedAgentId = [0, 2]
#     ropePartIndex = list(range(numAgent, numAgent + numKnots))
#     maxRopePartLength = 0.06
#     reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
#     numSimulationFrames=10
#     isTerminal= lambda state: False
#     distractorReshapeAction=ReshapeAction(5)
#     noiseMean = (0, 0)
#     noiseCov = [[1, 0], [0, 1]]
#     # x = np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0]
#     class LimitSpeed():
#         def __init__(self,entityMaxSpeed=None):
#             self.entityMaxSpeed = entityMaxSpeed

#         def __call__(self,entityNextVel):
#             if self.entityMaxSpeed is not None:
#                 speed = np.sqrt(np.square(entityNextVel[0]) + np.square(entityNextVel[1])) #
#             if speed > entityMaxSpeed:
#                 entityNextVel = entityNextVel / speed * entityMaxSpeed
#     limitSpeed = LimitSpeed(5)


#     noiseDistractorAction= lambda state:LimitSpeed(distractorReshapeAction(state)+np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0])

#     reshapeActionList = [ReshapeAction(5),ReshapeAction(5),ReshapeAction(masterForce),ReshapeAction(5)]
#     transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualizeMujoco,isTerminal, reshapeActionList)


#     sampleTrajectory = SampleExpTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)


#     observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID +distractorID, [], getPosFromAgentState, getVelFromAgentState)
#     observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]
#     print(reset())

#     initObsForParams = observe(reset())
#     obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
#     # print('24e',obsShape)
#     worldDim = 2
#     actionDim = worldDim * 2 + 1

#     layerWidth = [128, 128]

#     # ------------ model ------------------------
#     buildMADDPGModels = BuildMADDPGModels(actionDim, numAgent, obsShape)
#     modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgent)]

#     dataFolder = os.path.join(dirName, '..','..', 'data')
#     mainModelFolder = os.path.join(dataFolder,'model')
#     modelFolder = os.path.join(mainModelFolder, modelSaveName,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

#     fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

#     modelPaths = [os.path.join(modelFolder,  fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgent)]

#     [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

#     actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
#     policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]


#     trajList = []
#     expTrajList = []
#     for i in range(evalNum):
#         np.random.seed(i)
#         traj, expTraj = sampleTrajectory(policy)
#         trajList.append(list(traj))
#         expTrajList.append((list(expTraj)))

#     # print('save',saveTraj)
#     # saveTraj
#     if saveTraj:
#         # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

#         trajectoriesSaveDirectory= os.path.join(dataFolder,'trajectory',modelSaveName,'normal')
#         if not os.path.exists(trajectoriesSaveDirectory):
#             os.makedirs(trajectoriesSaveDirectory)

#         trajectorySaveExtension = '.pickle'
#         fixedParameters = {'damping': damping,'frictionloss':frictionloss,'masterForce':masterForce,'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
#         generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
#         trajectorySavePath = generateTrajectorySavePath({})
#         saveToPickle(trajList, trajectorySavePath)

#         expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectory', modelSaveName,'normal')
#         if not os.path.exists(expTrajectoriesSaveDirectory):
#             os.makedirs(expTrajectoriesSaveDirectory)

#         generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
#         exoTrajectorySavePath = generateExpTrajectorySavePath({})
#         saveToPickle(expTrajList, exoTrajectorySavePath)

#     # visualize
#     if visualizeTraj:

#         pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'normal','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

#         if not os.path.exists(pictureFolder):
#             os.makedirs(pictureFolder)
#         entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [masterColor] * numDistractor
#         render = Render(entitiesSizeList, entitiesColorList, numAgent,pictureFolder,saveImage, getPosFromAgentState)
#         trajToRender = np.concatenate(trajList)
#         render(trajToRender)
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
    manipulatedVariables['offset'] = [-0.5,0.0,0.5,1.0]
    manipulatedVariables['hideId'] = [3,4]
    manipulatedVariables2 = OrderedDict()
    # manipulatedVariables[''] = [0.5]#[0.0, 1.0]
    # manipulatedVariables[''] =[1.0]# [0.0, 0.2, 0.4]
    # manipulatedVariables['']=[0.0]#[0.0, 2.0]
    # manipulatedVariables2['offset'] = [0.5,-0.5]
    # manipulatedVariables2['hideId'] = [3,4]
    # manipulatedVariables['']=[3.0]
    damping = 0.5
    frictionloss = 0.0
    masterForce = 0.0
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



    # levelNames2 = list(manipulatedVariables2.keys())
    # levelValues2 = list(manipulatedVariables2.values())
    # modelIndex2 = pd.MultiIndex.from_product(levelValues2, names=levelNames2)
    # toSplitFrame2 = pd.DataFrame(index=modelIndex2)

    dataFolder = os.path.join(dirName, '..','..', 'data')
    # modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
    # trajectoryDirectory= os.path.join(dataFolder,'trajectory','noiseOffsetMasterForSelect')
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
