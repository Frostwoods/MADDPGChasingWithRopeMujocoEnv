import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import xmltodict
import mujoco_py as mujoco

import itertools as it
from collections import OrderedDict
import numpy as np
from env.multiAgentMujocoEnv import RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed

from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath，loadFromPickle
from src.functionTools.trajectory import SampleExpTrajectory
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
from src.visualize.visualizeMultiAgent import Render



wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
distractorColor = np.array([0.35, 0.85, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])




def generateSingleCondition(condition):
    debug = 0
    if debug:


        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 1
        maxTimeStep = 25

        maxEpisode = 60000
        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True
    else:

        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        damping = float(condition['damping'])
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])

        maxEpisode = 120000
        evaluateEpisode = 120000
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 1
        maxTimeStep = 25

        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = False
        makeVideo=False

    evalNum = 3
    maxRunningStepsToSample = 100
    modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
    print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))


    numAgent = numWolves + numSheeps + numMasters +  numDistractor
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3]

    wolfSize = 0.05
    sheepSize = 0.05
    masterSize = 0.05
    distractorSize = 0.05
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters + [distractorSize] * numDistractor


    entitiesMovableList = [True] * numAgent + [False] * numMasters

    killZone = 0.01
    isCollision = IsCollision(getPosFromAgentState, killZone)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardDistractor = RewardSheep(wolvesID+sheepsID+masterID, distractorID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))\
        + list(rewardMaster(state, action, nextState) )+ list(rewardDistractor(state, action, nextState))

    physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leasedAddDistractorForExp.xml')
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()

    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3,4]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)

    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3,4]
    keyNameList=[0,1]
    valueList=[[frictionloss,frictionloss]]*len(geomIds)
    frictionlossParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointFrictionlossProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@frictionloss')



    envXmlDict = xmltodict.parse(xml_string.strip())
    envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
    changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)

    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)

    numKnots = 9
    numAxis = (numKnots + numAgent) * 2
    qPosInit = (0, ) * numAxis
    qVelInit = (0, ) * numAxis
    qPosInitNoise = 0.4
    qVelInitNoise = 0
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(numAgent, numAgent + numKnots))
    maxRopePartLength = 0.06
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
    numSimulationFrames=10
    isTerminal= lambda state: False
    distractorReshapeAction=ReshapeAction(5)
    noiseMean = (0, 0)
    noiseCov = [[1, 0], [0, 1]]
    # x = np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0]
    class LimitSpeed():
        def __init__(self,entityMaxSpeed=None):
            self.entityMaxSpeed = entityMaxSpeed

        def __call__(self,entityNextVel):
            if self.entityMaxSpeed is not None:
                speed = np.sqrt(np.square(entityNextVel[0]) + np.square(entityNextVel[1])) #
            if speed > entityMaxSpeed:
                entityNextVel = entityNextVel / speed * entityMaxSpeed
    limitSpeed = LimitSpeed(5)


    noiseDistractorAction= lambda state:LimitSpeed(distractorReshapeAction(state)+np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0])

    reshapeActionList = [ReshapeAction(5),ReshapeAction(5),ReshapeAction(masterForce),ReshapeAction(5)]
    transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualizeMujoco,isTerminal, reshapeActionList)


    sampleTrajectory = SampleExpTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)


    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID +distractorID, [], getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]
    print(reset())

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    print('24e',obsShape)
    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgent, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgent)]

    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'model')
    modelFolder = os.path.join(mainModelFolder, modelSaveName,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPaths = [os.path.join(modelFolder,  fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgent)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]


    trajList = []
    expTrajList = []
    for i in range(evalNum):
        np.random.seed(i)
        traj, expTraj = sampleTrajectory(policy)
        trajList.append(list(traj))
        expTrajList.append((list(expTraj)))

    # print('save',saveTraj)
    # saveTraj
    if saveTraj:
        # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

        trajectoriesSaveDirectory= os.path.join(dataFolder,'trajectory',modelSaveName,'normal')
        if not os.path.exists(trajectoriesSaveDirectory):
            os.makedirs(trajectoriesSaveDirectory)

        trajectorySaveExtension = '.pickle'
        fixedParameters = {'damping': damping,'frictionloss':frictionloss,'masterForce':masterForce,'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
        generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
        trajectorySavePath = generateTrajectorySavePath({})
        saveToPickle(trajList, trajectorySavePath)

        expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectory', modelSaveName,'normal')
        if not os.path.exists(expTrajectoriesSaveDirectory):
            os.makedirs(expTrajectoriesSaveDirectory)

        generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
        exoTrajectorySavePath = generateExpTrajectorySavePath({})
        saveToPickle(expTrajList, exoTrajectorySavePath)

    # visualize
    if visualizeTraj:

        pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'normal','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

        if not os.path.exists(pictureFolder):
            os.makedirs(pictureFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [masterColor] * numDistractor
        render = Render(entitiesSizeList, entitiesColorList, numAgent,pictureFolder,saveImage, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajToRender)

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
def calculateChasingSubtlety(trajs):
    
    
    def calculateIncludedAngle(vector1,vector2):
        v1=complex(vector1[0],vector1[1])
        v2=complex(vector2[0],vector2[1])

        return np.angle(v1/v2)
    wolfSheepAngleList = []
    for traj in trajs:
        wolfSheepAngle=np.mean([calculateIncludedAngle() for state in  traj    ])
        wolfSheepAngleList.append(wolfSheepAngle)
    averageAngle = np.mean(wolfSheepAngleList)

    return averageAngle
    
class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = np.array([self.measurementFunction(trajectory) for trajectory in allTrajectories])
        measurementMean = np.mean(allMeasurements, axis = 0)
        measurementStd = np.std(allMeasurements, axis = 0)
        return pd.Series({'mean': measurementMean, 'std': measurementStd})

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.5,1.0]#[0.0, 1.0]
    manipulatedVariables['frictionloss'] =[0.1,0.2]# [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce']=[0.5,1.0]#[0.0, 2.0]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    evalNum=3
    evaluateEpisode=200000

    for condition in conditions:
        print(condition)
        # generateSingleCondition(condition)
        try:
            generateSingleCondition(condition)
        except:
            continue


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    evalNum=3

    dataFolder = os.path.join(dirName, '..','..', 'data')
    modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
    trajectoryDirectory= os.path.join(dataFolder,'trajectory',modelSaveName,'normal')  
    trajectoryExtension = '.pickle'

    trajectoryFixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
    
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: calculateChasingSubtlety(trajectory)[0]
    
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)    
    
    print(statisticsDf)
    
    
    # fig = plt.figure()
    # numRows = len(manipulatedVariables['miniBatchSize'])
    # numColumns = len(manipulatedVariables['depth'])
    # plotCounter = 1

    # for miniBatchSize, grp in statisticsDf.groupby('miniBatchSize'):
    #     grp.index = grp.index.droplevel('miniBatchSize')

    #     for depth, group in grp.groupby('depth'):
    #         group.index = group.index.droplevel('depth')

    #         axForDraw = fig.add_subplot(numRows,numColumns,plotCounter)
    #         if plotCounter % numColumns == 1:
    #             axForDraw.set_ylabel('miniBatchSize: {}'.format(miniBatchSize))
    #         if plotCounter <= numColumns:
    #             axForDraw.set_title('depth: {}'.format(depth))
    #         axForDraw.set_ylim(-1, 1.3)

    #         # plt.ylabel('Distance between optimal and actual next position of sheep')
    #         drawPerformanceLine(group, axForDraw)
    #         trainStepLevels = statisticsDf.index.get_level_values('trainSteps').values
    #         axForDraw.plot(trainStepLevels, [1.18]*len(trainStepLevels), label='mctsTrainData')
    #         plotCounter += 1
    # # plt.supertitle('Sheep')

    # plt.legend(loc='best')
    # plt.show()
if __name__ == '__main__':
    main()
