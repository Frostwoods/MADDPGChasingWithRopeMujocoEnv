import os
import sys
import glob
DIRNAME = os.path.dirname(__file__)

dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName))
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from collections import OrderedDict
import xmltodict
import mujoco_py as mujoco

from src.maddpg.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from src.RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from src.functionTools.loadSaveModel import restoreVariables,GetSavePath
from env.multiAgentMujocoEnv import RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
# class SampleAllState():
#     def __init__(self,reset):

#         self.reset = reset
#     def __call__(self,wolfPos,sheepPos):
#         initState = self.reset(,wolfPos)
class FixResetUniformWithoutXPosForLeashed:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, tiedAgentIndex, ropePartIndex, maxRopePartLength, qPosInitNoise=0, qVelInitNoise=0):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.tiedBasePosAgentIndex, self.tiedFollowPosAgentIndex = tiedAgentIndex
        self.numRopePart = len(ropePartIndex)
        self.maxRopePartLength = maxRopePartLength
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)

    def __call__(self,wolfPos,sheepPos):
        sheepId = 1
        wolfId = 0
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qPos[2 * sheepId : 2*(sheepId+1)] = sheepPos
        qPos[2 * wolfId : 2*(wolfId+1)] = wolfPos
        tiedBasePos = qPos[self.numJointEachSite * self.tiedBasePosAgentIndex: self.numJointEachSite * (self.tiedBasePosAgentIndex + 1)]
        sampledRopeLength = np.random.uniform(low = 0, high = self.numRopePart * self.maxRopePartLength)
        sampledPartLength = np.arange(sampledRopeLength/(self.numRopePart + 1), sampledRopeLength, sampledRopeLength/(self.numRopePart + 1))[:self.numRopePart]
        theta = np.random.uniform(low = 0, high = math.pi)

        tiedFollowPosAgentPos = tiedBasePos + np.array([sampledRopeLength * np.cos(theta), sampledRopeLength * np.sin(theta)])
        qPos[self.numJointEachSite * self.tiedFollowPosAgentIndex : self.numJointEachSite * (self.tiedFollowPosAgentIndex + 1)] = tiedFollowPosAgentPos
        ropePartPos = np.array(list(zip(sampledPartLength * np.cos(theta), sampledPartLength * np.sin(theta)))) + tiedBasePos
        qPos[-self.numJointEachSite * self.numRopePart : ] = np.concatenate(ropePartPos)

        qVelSampled = np.concatenate([np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel - self.numRopePart * self.numJointEachSite),np.zeros(self.numRopePart * self.numJointEachSite)])
        qVel = self.qVelInit + qVelSampled

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState
class DrawGird():
    def __init__(self):
        pass
    def __call__(self,ax,wolfPos):
        ax.grid(color='r',linestyle='--')
        plt.scatter(wolfPos[0], wolfPos[1], s=0.05, c=[255,0,0], alpha=0.5)
class DrawSingleArrow():
    def __init__(self):
        pass
    def __call__(self,ax,sheepPos,action):
        plt.scatter(sheepPos[0], sheepPos[1], s=0.05, c=[255,0,0], alpha=0.5)
        ax.arrow(sheepPos[0], sheepPos[1], action[0], action[1], width=0.01,length_includes_head=True,head_width=0.25,head_length=1,fc='r',ec='b')
 

class DrawSingleChasingMap():
    def __init__(self,sheepFixPosList,sampleNums,sampleAllState,sheepPolicy,drawGird,drawSingleArrow):
        self.sheepPolicy = sheepPolicy
        self.sheepFixPosList = sheepFixPosList
        self.sampleNums = sampleNums
        self.sampleAllState = sampleAllState
        self.drawGird = drawGird
        self.drawSingleArrow = drawSingleArrow
        # self.sheepPolicy = sheepPolicy
    def __call__(self,wolfPos,ax):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        self.drawGird(ax,wolfPos)
        for sheepPos in self.sheepFixPosList:
            allAction = [self.sheepPolicy(self.sampleAllState(wolfPos,sheepPos))]
            self.drawSingleArrow(ax,sheepPos,np.mean(allAction))

        pass
if __name__ == '__main__':
    # manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.5]
    # manipulatedVariables['frictionloss'] = [1.0]
    # manipulatedVariables['masterForce'] = [1.0]
    # manipulatedVariables['killZone'] = [2.0, 4.0]
    # manipulatedVariables['ropePunishWeight'] = [0.3, 0.5]
    # manipulatedVariables['ropeLength'] = [0.06] #ssr-1,Xp = 0.06; ssr-3 =0.09
    # manipulatedVariables['masterMass'] = [1.0] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    # manipulatedVariables['offset'] = [0.0]
    
    # levelNames = list(manipulatedVariables.keys())
    # levelValues = list(manipulatedVariables.values())
    # modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    # toSplitFrame = pd.DataFrame(index=modelIndex)

    damping = 0.5
    frictionloss = 1.0
    masterForce = 1.0
    offset = 0.0
    killZoneRatio = 2.0#4.0
    ropePunishWeight = 0.3
    ropeLength = 0.06
    masterMass = 1.0
    dt = 0.02
    offsetFrame = int (offset/dt)

    maxEpisode = 120000
    evaluateEpisode = 120000
    numWolves = 1
    numSheeps = 1
    numMasters = 1
    numDistractor = 2
    maxTimeStep = 25
    maxTimeStep = 25

    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3,4]
    numWolves = 1
    numSheeps = 1
    numMasters = 1
    numDistractor = 2
    hideIdList =  distractorID + sheepsID
    numAgent=5

    physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leased2Distractor_masterMass={}_ropeLength={}.xml'.format(masterMass,ropeLength))
    print('loadEnv:{}'.format(physicsDynamicsPath))
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()

    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3,4,5]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3,4,5]
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
    qPosInitNoise = 0.6
    qVelInitNoise = 0
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(numAgent, numAgent+numKnots))
    maxRopePartLength = ropeLength

    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
    fixReset = FixResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)  

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
    modelFolder = os.path.join(mainModelFolder, 'modelSaveName','damping={}_frictionloss={}_killZoneRatio{}_masterForce={}_masterMass={}_ropeLength={}_ropePunishWeight={}'.format(damping,frictionloss,killZoneRatio,masterForce,masterMass,ropeLength,ropePunishWeight))
    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPaths = [os.path.join(modelFolder,  fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgent)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]
    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    # policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
    sheepPolicy = lambda allAgentsStates: actOneStepOneModel(modelsList[sheepsID[0]], observe(allAgentsStates))



    sheepPosRange = [-0.8,-0.4,0.0,0.4,0.8]
    sheepFixPosList = it.product(sheepPosRange,sheepPosRange)
    sampleNums = 10

    drawGird = DrawGird()
    drawSingleArrow = DrawSingleArrow()
    drawSingleChasingMap = DrawSingleChasingMap(sheepFixPosList,sampleNums,fixReset,sheepPolicy,drawGird,drawSingleArrow)




    posValue = OrderedDict()
    posValue['wolfXPos'] = [-0.50,0.00,0.50]
    posValue['wolfYPos'] = [-0.50,0.00,0.50]

    fig = plt.figure()
    rowName = 'wolfXPos'
    columnName = 'wolfYPos'
    numRows = len(posValue[rowName])
    numColumns = len(posValue[columnName])
    plotCounter = 1

    for xPos in posValue[rowName]:
        for yPos in posValue[columnName]:
            axForDraw = fig.add_subplot(numRows,numColumns,plotCounter)
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel(rowName+': {}'.format(xPos))
            if plotCounter <= numColumns:
                axForDraw.set_title(columnName+': {}'.format(yPos))
            drawSingleChasingMap([xPos,yPos])
            plotCounter += 1
    plt.suptitle('chasingMap')