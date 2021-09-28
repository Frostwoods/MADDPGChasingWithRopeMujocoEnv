import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import json
# import xmltodict
# import mujoco_py as mujoco
import math

from src.maddpg.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from src.RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from src.functionTools.loadSaveModel import saveVariables
# from env.multiAgentEnv import GetActionCost,RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetMultiAgentNewtonChasing
from env.multiAgentEnv import GetActionCost,RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, getVelFromAgentState,PunishForOutOfBound, StayInBoundaryByReflectVelocity,ResetMultiAgentNewtonChasing,TransitMultiAgentChasingForExp, ReshapeAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, IntegrateState, getPosFromAgentState,getVelFromAgentState,ResetMultiAgentChasingWithVariousSheep,ReshapeActionVariousForce,TransitMultiAgentChasingForExpVariousForce
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty


# fixed training parameter,
maxEpisode = 120000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# 7.13 add action cost
# 7.29 constant sheep bonus = 30
# 8.9 add degree of individuality
# TODO: 8.23 changed sheep punishment to 10 to test consistency with source performance
# 8.26 changed constant sheep bonus = 10

def main():
    debug = 0
    if debug:
        numWolves = 2
        numSheeps = 1
        numBlocks = 0
        saveAllmodels = 0
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.3
        individualRewardWolf = 0
        costActionRatio = 0.0

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        sheepWolfForceRatio = float(condition['sheepWolfForceRatio'])
        killZoneRatio = float(condition['killZoneRatio'])

        numBlocks = 0

        wolfForce = 1 
        sheepForce = wolfForce * sheepWolfForceRatio

        maxTimeStep = 100# int(condition['maxTimeStep'])
        sheepSpeedMultiplier =1.3 #float(condition['sheepSpeedMultiplier'])
        individualRewardWolf =0 #float(condition['individualRewardWolf'])
        costActionRatio = 0#float(condition['costActionRatio'])

        saveAllmodels = 1

    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}, save all models: {}".
          format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf, str(saveAllmodels)))


    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'model')
    modelFolder = os.path.join(mainModelFolder, 'originEnv9.16','indvidulReward={}_sheepWolfForceRatio={}_killZoneRatio={}'.format(individualRewardWolf,sheepWolfForceRatio,killZoneRatio),'{} wolves, {} sheep, {} blocks'.format(numWolves, numSheeps, numBlocks))

    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.0
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1 #!!
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10 # originalPaper = 10*3
    baselineKillzone = wolfSize + sheepSize
    addKillZone = baselineKillzone * (killZoneRatio-1)
    isCollision = IsCollision(getPosFromAgentState,addKillZone)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound, collisionPunishment = collisionReward) # TODO collisionReward = collisionPunishment

    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
    # reshapeAction = ReshapeAction()
    # getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    # getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    # rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    # rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    
    minDistanceForReborn = 10
    # numPlayers = 2
    # gridSize = 60
    # reset0 = ResetMultiAgentNewtonChasing(gridSize, numWolves, minDistanceForReborn)
    reset0 =  ResetMultiAgentChasingWithVariousSheep(numWolves, numBlocks)
    reset = lambda :reset0(numSheeps)
    # reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    
    # stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity( [0, gridSize - 1], [0, gridSize - 1])
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity( [-1,1], [-1, 1])
    def checkBoudary(agentState):
        newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState),getVelFromAgentState(agentState))
        return newState
    checkAllAgents = lambda states:[checkBoudary(agentState) for agentState in states]
    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,  getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    # transit = TransitMultiAgentChasingForExp(reshapeAction, applyActionForce, applyEnvironForce, integrateState,checkAllAgents)
    reshapeAction = ReshapeActionVariousForce()
    expTransit = TransitMultiAgentChasingForExpVariousForce(reshapeAction, reshapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)
    transit = lambda state,actions:expTransit(state,actions[0:numWolves],actions[-numSheeps:],wolfForce,sheepForce)

    isTerminal = lambda state: [False]* numAgents
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

#------------ models ------------------------

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]

    sampleOneStep = SampleOneStep(transit, rewardFunc)
    runTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe = observe)

    runEpisode = RunEpisode(reset, runTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 2000
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}individ{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep,   individualRewardWolf)

    # folderName = 'maddpg_10reward_full'
    modelPath = os.path.join(modelFolder, fileName)
    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath + str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList = maddpg(replayBuffer)


if __name__ == '__main__':
    main()



