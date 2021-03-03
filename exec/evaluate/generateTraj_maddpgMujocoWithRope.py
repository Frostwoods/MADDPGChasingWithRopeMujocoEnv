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

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables
from src.functionTools.trajectory import SampleTrajectory
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
from src.visualize.visualizeMultiAgent import Render



wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000



def generateSingleCondition(condition):
    debug = 0
    if debug:


        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numWolves = 1
        numSheeps = 1
        numMasters = 1
        maxTimeStep = 25

        saveTraj=False
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
        evaluateEpisode = 60000
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        maxTimeStep = 25

        saveTraj=False
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=False

    print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))


    numAgents = numWolves + numSheeps+numMasters
    numEntities = numAgents + numMasters
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]

    wolfSize = 0.075
    sheepSize = 0.05
    masterSize = 0.075
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None


    entitiesMovableList = [True] * numAgents + [False] * numMasters
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)


    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))+list(rewardMaster(state, action, nextState))

    dirName = os.path.dirname(__file__)
    # physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','rope','leased.xml')

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


    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xml', 'leasedNew2.xml')

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
    qPosInitNoise = 0.4
    qVelInitNoise = 0
    numAgent = 2
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(3, 12))
    maxRopePartLength = 0.06
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
    numSimulationFrames=10
    isTerminal= lambda state: False
    reshapeActionList = [ReshapeAction(5),ReshapeAction(5),ReshapeAction(masterForce)]
    transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualizeMujoco,isTerminal, reshapeActionList)

    maxRunningStepsToSample = 100
    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, masterID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'model')
    modelFolder = os.path.join(mainModelFolder, 'MADDPGMujocoEnvWithRope','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPaths = [os.path.join(modelFolder,  fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]


    trajList = []
    numTrajToSample = 5
    for i in range(numTrajToSample):
        np.random.seed(i)
        traj = sampleTrajectory(policy)
        trajList.append(list(traj))

    # saveTraj
    if saveTraj:
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

        trajFolder= os.path.join(dataFolder,'trajectory')
        trajSavePath = os.path.join(trajFolder,'MADDPGMujocoEnvWithRope', trajFileName)
        saveToPickle(trajList, trajSavePath)


    # visualize
    if visualizeTraj:

        demoFolder = os.path.join(dataFolder, 'demo', 'MADDPGMujocoEnvWithRope','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

        if not os.path.exists(demoFolder):
            os.makedirs(demoFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters
        render = Render(entitiesSizeList, entitiesColorList, numAgents,demoFolder,saveImage, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajToRender)


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce']=[0.0, 2.0]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        print(condition)
        try:
            generateSingleCondition(condition)
        except:
            continue

if __name__ == '__main__':
    main()
