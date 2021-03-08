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
import xmltodict
import mujoco_py as mujoco
import math

from src.maddpg.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from src.RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from src.functionTools.loadSaveModel import saveVariables
from env.multiAgentMujocoEnv import RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty


# fixed training parameters
maxEpisode = 120000#150000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# arguments: numWolves numSheeps numMasters saveAllmodels = True or False

def main():
    debug = 1
    if debug:

        damping=0.0
        frictionloss=0.4
        masterForce=1.0


        numWolves = 1
        numSheeps = 1
        numMasters = 1
        maxTimeStep = 25
        visualize=False
        saveAllmodels = True

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        damping = float(condition['damping'])
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])

        maxTimeStep = 25
        visualize=False
        saveAllmodels = True
    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps,  save all models: {}".
          format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep,  str(saveAllmodels)))
    print(damping,frictionloss,masterForce)

    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'model')
    modelFolder = os.path.join(mainModelFolder, 'MADDPGMujocoEnvWithRope','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)


    numAgents = numWolves + numSheeps+numMasters
    numEntities = numAgents + numMasters
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3]
    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.075
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numMasters



    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)


    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))+list(rewardMaster(state, action, nextState))


    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)

    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3]
    keyNameList=[0,1]
    valueList=[[frictionloss,frictionloss]]*len(geomIds)
    frictionlossParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointFrictionlossProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@frictionloss')

    physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leasedNew2.xml')
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()


    envXmlDict = xmltodict.parse(xml_string.strip())
    envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
    changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)



    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)


    qPosInit = (0, ) * 24
    qVelInit = (0, ) * 24
    qPosInitNoise = 0.6
    qVelInitNoise = 0
    numAgent = 3
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(3, 12))
    maxRopePartLength = 0.06
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

    numSimulationFrames=10
    isTerminal= lambda state: False
    reshapeActionList = [ReshapeAction(5),ReshapeAction(5),ReshapeAction(masterForce)]
    transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualize,isTerminal, reshapeActionList)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, masterID, getPosFromAgentState,getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
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
    runDDPGTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe = observe)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 1000
    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPath = os.path.join(modelFolder, fileName)

    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = maddpg(replayBuffer)





if __name__ == '__main__':
    main()


