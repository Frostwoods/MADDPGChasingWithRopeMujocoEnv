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
# from env.multiAgentMujocoEnv import TransitionFunctionWithoutXPosForExp, RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    # getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed

from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath
from src.functionTools.trajectory import SampleExpTrajectory,SampleExpTrajectoryWithAllFrames
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
from src.visualize.visualizeMultiAgent import Render

from env.multiAgentEnv import GetActionCost, RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, getVelFromAgentState, PunishForOutOfBound, StayInBoundaryByReflectVelocity, ResetMultiAgentNewtonChasing, TransitMultiAgentChasingForExp, ReshapeAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, IntegrateState, getPosFromAgentState, getVelFromAgentState, ResetMultiAgentChasingWithVariousSheep, ReshapeActionVariousForce, TransitMultiAgentChasingForExpVariousForce
from src.functionTools.editEnvXml import transferNumberListToStr, MakePropertyList, changeJointProperty


wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
distractorColor = np.array([0.35, 0.85, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])




def generateSingleCondition(condition):
    debug = 1
    if debug:

        condition = {}
        damping=2.0
        frictionloss=0.0
        masterForce=1.0


        maxTimeStep = 25

        maxEpisode = 60000
        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True



        chasePairList = [[2, 1], [1, 1]]
        # chasePairIdList =
        chasePairIdList = [[[0, 1], [3]], [[2], [4]]]

        numWolves = sum([pair[0] for pair in chasePairList])
        numSheeps = sum([pair[1] for pair in chasePairList])

        # numWolves = 4
        # numSheeps = 2
        numDistractors = 1
        numBlocks = 0

        condition['numWolves'] = numWolves
        condition['numSheeps'] = numSheeps
        condition['numDistractors'] = numDistractors 
        condition['numBlocks'] =  numBlocks

        saveAllmodels = 0
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.3
        individualRewardWolf = 0
        killZoneRatio = 1
        sizeRatio = 1
        maxRange = 1.0
        sheepWolfForceRatio = 1.3
        wolfForce = 1
        sheepForce = wolfForce * sheepWolfForceRatio

        maxTimeStep = 100  # int(condition['maxTimeStep'])
        sheepSpeedMultiplier = 1.3  # float(condition['sheepSpeedMultiplier'])
        costActionRatio = 0  # float(condition['costActionRatio'])

        saveAllmodels = 1

    else:


        damping=2.0
        frictionloss=0.0
        masterForce=1.0


        maxTimeStep = 25

        maxEpisode = 60000
        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True



        chasePairList = [[2, 1], [1, 1]]
        # chasePairIdList =
        chasePairIdList = [[[0, 1], [3]], [[2], [4]]]

        numWolves = sum([pair[0] for pair in chasePairList])
        numSheeps = sum([pair[1] for pair in chasePairList])

        # numWolves = 4
        # numSheeps = 2
        numDistractors = 1
        numBlocks = 0

        saveAllmodels = 0
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.3
        individualRewardWolf = 0
        killZoneRatio = 1
        sizeRatio = 1
        maxRange = 1.0
        sheepWolfForceRatio = 1.3
        wolfForce = 1
        sheepForce = wolfForce * sheepWolfForceRatio

        maxTimeStep = 100  # int(condition['maxTimeStep'])
        sheepSpeedMultiplier = 1.3  # float(condition['sheepSpeedMultiplier'])
        costActionRatio = 0  # float(condition['costActionRatio'])

        saveAllmodels = 1


        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        damping = float(condition['damping'])
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])
        distractorNoise = float(condition['distractorNoise'])
        offset = float(condition['offset'])

        dt = 0.02
        offsetFrame = int (offset/dt)

        maxEpisode = 120000
        evaluateEpisode = 120000
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 2
        maxTimeStep = 25

        noiseDistractor=True
        if noiseDistractor:
            distractorNoise = float(condition['distractorNoise'])

        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=False

    evalNum = 20
    maxRunningStepsToSample = 100
    modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAdd2Distractors'
    print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))

    dataFolder = os.path.join(dirName, '..', '..', 'data')
    mainModelFolder = os.path.join(dataFolder, 'model')
    modelFolder = os.path.join(mainModelFolder, 'GroupChasing', 'indvidulReward={}_sheepWolfForceRatio={}_killZoneRatio={}'.format(individualRewardWolf, sheepWolfForceRatio, killZoneRatio), '{} wolves, {} sheep,{}distractors, {} blocks'.format(numWolves, numSheeps, numDistractors, numBlocks))

    # hideIdList = sheepsID + distractorID


    numAgents = numWolves + numSheeps + numDistractors
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numWolves + numSheeps))
    distractorsID = list(range(numWolves + numSheeps, numAgents))
    blocksID = list(range(numAgents, numEntities))


    wolfSize = 0.05 * sizeRatio
    sheepSize = 0.05 * sizeRatio
    distractorSize = 0.05 * sizeRatio
    blockSize = 0.5
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [distractorSize] * numDistractors + [blockSize] * numBlocks

    wolfMaxSpeed = 1  # !!
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [sheepMaxSpeed] * numDistractors + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10  # originalPaper = 10*3
    baselineKillzone = wolfSize + sheepSize
    addKillZone = baselineKillzone * (killZoneRatio - 1)
    isCollision = IsCollision(getPosFromAgentState, addKillZone)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheepList = []
    rewardWolfList = []

    for chassePair in chasePairIdList:
        ingroupWolvesID = chassePair[0]
        ingroupSheepID = chassePair[1]
        rewardIngroupSheep = RewardSheep(ingroupWolvesID, ingroupSheepID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisionPunishment=collisionReward)  # TODO collisionReward = collisionPunishment

        rewardIngroupWolf = RewardWolf(ingroupWolvesID, ingroupSheepID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
        rewardSheepList.append(rewardIngroupSheep)
        rewardWolfList.append(rewardIngroupWolf)
        # rewardSheepList = rewardSheepList + rewardIngroupSheep
        # rewardWolfList = rewardWolfList + rewardIngroupWolf

    if numDistractors > 0:
        rewardDistractiors = RewardSheep(wolvesID + sheepsID, distractorsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisionPunishment=collisionReward)

    def rewardWolf(state, action, nextState):

        return [singleReward(state, action, nextState) for singleReward in rewardWolfList]

    def rewardSheep(state, action, nextState):
        return [singleReward(state, action, nextState) for singleReward in rewardSheepList]
    # reshapeAction = ReshapeAction()
    # getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    # getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    # rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    # rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    def rewardFunc(state, action, nextState):

        if numDistractors > 0:
            # allreward = list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState)) + list(rewardDistractiors(state, action, nextState))
            allreward = []
            [allreward.extend(reward) if isinstance(reward, list) else allreward.append(reward) for reward in list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState)) + list(rewardDistractiors(state, action, nextState))]

            return allreward

        else:
            # print(list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState)))
            allreward = []
            [allreward.extend(reward) if isinstance(reward, list) else allreward.append(reward) for reward in list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))]
            return allreward
    # minDistanceForReborn = 10
    # numPlayers = 2
    # gridSize = 60
    # reset0 = ResetMultiAgentNewtonChasing(gridSize, numWolves, minDistanceForReborn)
    reset0 = ResetMultiAgentChasingWithVariousSheep(numWolves, numBlocks, maxRange)

    def reset(): return reset0(numSheeps + numDistractors)
    # reset = ResetMultiAgentChasing(numAgents, numBlocks)

    def observeOneAgent(agentID): return Observe(agentID, wolvesID, sheepsID + distractorsID, blocksID, getPosFromAgentState, getVelFromAgentState)

    def observe(state): return [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    # stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity( [0, gridSize - 1], [0, gridSize - 1])
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-maxRange, maxRange], [-maxRange, maxRange])

    def checkBoudary(agentState):
        newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
        return newState

    def checkAllAgents(states): return [checkBoudary(agentState) for agentState in states]
    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID + distractorsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    # transit = TransitMultiAgentChasingForExp(reshapeAction, applyActionForce, applyEnvironForce, integrateState,checkAllAgents)
    reshapeAction = ReshapeActionVariousForce()
    expTransit = TransitMultiAgentChasingForExpVariousForce(reshapeAction, reshapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)

    def transit(state, actions): return expTransit(state, actions[0:numWolves], actions[numWolves:], wolfForce, sheepForce)

    def isTerminal(state): return [False] * numAgents
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    sampleTrajectory = SampleExpTrajectoryWithAllFrames(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)



    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    print('24e',obsShape)
    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    fileName = "maddpg{}wolves{}sheep{}distracors{}blocks{}episodes{}individ{}_agent".format(
        numWolves, numSheeps, numDistractors, numBlocks, maxEpisode, maxTimeStep, individualRewardWolf)

    modelPaths = [os.path.join(modelFolder,  fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
    hideIdList=[1]
    for hideId in hideIdList:
        agentList = list(range(numAgents))
        # del(agentList[hideId])
        trajList = []
        expTrajList = []
        newTrajList = []
        for _ in range(evalNum):
            # np.random.seed(i)
            traj, expTraj = sampleTrajectory(policy)
            trajList.append(list(traj))
            expTrajList.append((list(expTraj)))
        for i,traj in enumerate(expTrajList):

            newTraj = [[state[agentId] for agentId in agentList]  for state in traj]
            offsetTraj =  [[newTraj[index][0],newTraj[index+offsetFrame][1],newTraj[index+offsetFrame][2],newTraj[index+offsetFrame][3]]    for index in range(len(newTraj)-offsetFrame)]
            newTrajList.append(offsetTraj)
        
        # print('save',saveTraj)
        # saveTraj
        if saveTraj:
            # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

            trajectoriesSaveDirectory= os.path.join(dataFolder,'trajectory',modelSaveName,'noise')
            if not os.path.exists(trajectoriesSaveDirectory):
                os.makedirs(trajectoriesSaveDirectory)

            trajectorySaveExtension = '.pickle'
            fixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
            generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            trajectorySavePath = generateTrajectorySavePath(condition)
            saveToPickle(trajList, trajectorySavePath)

            expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectoryHideOneAgent', modelSaveName,'noise')
            if not os.path.exists(expTrajectoriesSaveDirectory):
                os.makedirs(expTrajectoriesSaveDirectory)

            generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            expTrajectorySavePath = generateExpTrajectorySavePath(condition)
            saveToPickle(newTrajList, expTrajectorySavePath)

        # visualize
        # if visualizeTraj:

        #     pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'normal','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

        #     if not os.path.exists(pictureFolder):
        #         os.makedirs(pictureFolder)
        #     entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [distractorColor] * numDistractor
        #     render = Render(entitiesSizeList, entitiesColorList, numAgent,pictureFolder,saveImage, getPosFromAgentState)
        #     trajToRender = np.concatenate(expTrajList)
        #     print(np.size(trajToRender,0))
        #     render(trajToRender)


def main():

    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.0,0.5]#[0.0, 1.0]
    manipulatedVariables['frictionloss'] =[1.0]# [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce']=[0.0, 1.0]#[0.0, 2.0]
    manipulatedVariables['offset'] = [0.0, 1.0]
    manipulatedVariables['distractorNoise']=[3.0]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        # print(condition)
        generateSingleCondition(condition)
        # try:
            # generateSingleCondition(condition)
        # except:
            # continue

if __name__ == '__main__':
    main()
