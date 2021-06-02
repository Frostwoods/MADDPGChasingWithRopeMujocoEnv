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
#     getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed
from env.multiAgentEnv import GetActionCost,RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, getVelFromAgentState,PunishForOutOfBound, StayInBoundaryByReflectVelocity,ResetMultiAgentNewtonChasing,TransitMultiAgentChasingForExp, ReshapeAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, IntegrateState, getPosFromAgentState,getVelFromAgentState
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath
from src.functionTools.trajectory import SampleExpTrajectory,SampleExpTrajectoryWithAllFrames
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

        numWolves = 2
        numSheeps = 1
        numBlocks = 0
        saveAllmodels = 0
        maxTimeStep = 75
        sheepSpeedMultiplier = 1
        individualRewardWolf = 0
        costActionRatio = 0.0

        # damping=2.0
        # frictionloss=0.0
        # masterForce=1.0

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
        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        # masterForce = float(condition['masterForce'])
        print(sys.argv)
        # condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = 0

        maxEpisode = 60000
        evaluateEpisode = 60000
        # numWolves = 1
        # numSheeps = 1
        # numMasters = 1
        # numDistractor = 2

        maxTimeStep = 75# int(condition['maxTimeStep'])
        sheepSpeedMultiplier =1 #float(condition['sheepSpeedMultiplier'])
        individualRewardWolf =0 #float(condition['individualRewardWolf'])
        costActionRatio = 0#float(condition['costActionRatio'])

        # saveAllmodels = 1
        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=False

    evalNum = 3
    maxRunningStepsToSample = 100

    # modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAdd2DistractorsWithRopePunish'
    # modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAdd2Distractors'
    # print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))


    numAgent = numWolves + numSheeps
    numEntities = numAgent + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgent))
    blocksID = list(range(numAgent, numEntities))

    wolfSize = 0.75
    sheepSize = 0.75
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1000
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1000
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgent + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment = collisionReward) # TODO collisionReward = collisionPunishment

    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    
    minDistanceForReborn = 10
    # numPlayers = 2
    gridSize = 60
    reset0 = ResetMultiAgentNewtonChasing(gridSize, numWolves, minDistanceForReborn)
    reset = lambda :reset0(numSheeps)
    # reset = ResetMultiAgentChasing(numAgent, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]
    
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity( [0, gridSize - 1], [0, gridSize - 1])
    def checkBoudary(agentState):
        newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState),getVelFromAgentState(agentState))
        return newState
    checkAllAgents = lambda states:[checkBoudary(agentState) for agentState in states]
    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,  getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasingForExp(reshapeAction, applyActionForce, applyEnvironForce, integrateState,checkAllAgents)

    isTerminal = lambda state: False


    sampleTrajectory = SampleExpTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)


    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]

    # print(reset())

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    print('24e',obsShape)
    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgent, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgent)]

    dataFolder = os.path.join(dirName, '..','..', 'data','fakeNewtonEnv')
    mainModelFolder = os.path.join(dataFolder,'model','fakeNewton')
    # modelFolder = os.path.join(mainModelFolder, modelSaveName,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    modelFolder = os.path.join(mainModelFolder,'{} wolves, {} sheep, {} blocks'.format(numWolves, numSheeps, numBlocks))

    # fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individualRewardWolf)

    modelPaths = [os.path.join(modelFolder, fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgent)]
    print(modelPaths)
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
    # if saveTraj:
        # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

        # trajectoriesSaveDirectory= os.path.join(dataFolder,'trajectory','normal')
        # if not os.path.exists(trajectoriesSaveDirectory):
        #     os.makedirs(trajectoriesSaveDirectory)

        # trajectorySaveExtension = '.pickle'
        # fixedParameters = {'numWolves': numWolves,'numSheeps':numSheeps,'numBlocks':numBlocks,'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
        # generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
        # trajectorySavePath = generateTrajectorySavePath({})
        # saveToPickle(trajList, trajectorySavePath)

        # expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectory', modelSaveName,'normal')
        # if not os.path.exists(expTrajectoriesSaveDirectory):
        #     os.makedirs(expTrajectoriesSaveDirectory)

        # generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
        # expTrajectorySavePath = generateExpTrajectorySavePath({})
        # saveToPickle(expTrajList, expTrajectorySavePath)

    # visualize
    if visualizeTraj:

        pictureFolder = os.path.join(dataFolder, 'demo', 'normal','numWolves={}_numSheeps={}_numBlocks={}'.format(numWolves,numSheeps,numBlocks))

        if not os.path.exists(pictureFolder):
            os.makedirs(pictureFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps 
        render = Render(entitiesSizeList, entitiesColorList, numAgent,pictureFolder,saveImage, getPosFromAgentState)
        trajToRender = np.concatenate(expTrajList)
        # print(np.size(trajToRender,1))
        render(trajToRender)


def main():

    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2]#[0.0, 1.0]
    manipulatedVariables['numSheeps'] =[2]# [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce']=[0.0, 1.0]#[0.0, 2.0]
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
