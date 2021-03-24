import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(sys.path[0])
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import random
import xmltodict
import mujoco_py as mujoco

import itertools as it
from collections import OrderedDict
import numpy as np
from env.multiAgentMujocoEnv import RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed

from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath,loadFromPickle
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
        offsetFrame=5
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
        offsetFrame = int(condition['offsetFrame'])
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
        visualizeTraj = True
        makeVideo=False

    evalNum = 3
    maxRunningStepsToSample = 100
    modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
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

    dataFolder = os.path.join(dirName, '..','..', 'data')
    expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectory', modelSaveName)
    # if not os.path.exists(expTrajectoriesSaveDirectory):
    #     os.makedirs(expTrajectoriesSaveDirectory)
    trajectorySaveExtension = '.pickle'
    fixedParameters = {'damping': damping,'frictionloss':frictionloss,'masterForce':masterForce,'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
    generateExpTrajectoryLoadPath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    expTrajectoryLoadPath = generateExpTrajectoryLoadPath({})
    originalTrajList=loadFromPickle(expTrajectoryLoadPath)
    newTrajList=[]
    for i,traj in enumerate(originalTrajList):
        # offsetFrame=5
        newTraj = [[traj[i][0],traj[i+offsetFrame][1],traj[i+offsetFrame][2],traj[i+offsetFrame][3]]    for index in range(len(traj)-offsetFrame)]
        # newTraj = [[replaceState[0],originalState[1],originalState[2],originalState[3]] for originalState,replaceState in zip (traj,originalTrajList[j])]
        newTrajList.append(newTraj)

    expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectoryOffset', modelSaveName)
    generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    expTrajectorySavePath = generateExpTrajectorySavePath({})
    saveToPickle(newTrajList, expTrajectorySavePath)    

    if visualizeTraj:

        pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'damping={}_frictionloss={}_masterForce={}_offsetFrame={}'.format(damping,frictionloss,masterForce,offsetFrame))

        if not os.path.exists(pictureFolder):
            os.makedirs(pictureFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [masterColor] * numDistractor
        render = Render(entitiesSizeList, entitiesColorList, numAgent,pictureFolder,saveImage, getPosFromAgentState)
        trajToRender = np.concatenate(newTrajList)
        render(trajToRender)

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.0,1.0]#[0.0, 1.0]
    manipulatedVariables['frictionloss'] =[0.0,0.2]# [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce']=[0.0,1.0]#[0.0, 2.0]
    manipulatedVariables['offsetFrame']=[5]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        print(condition)
        # generateSingleCondition(condition)
        try:
            generateSingleCondition(condition)
        except:
            continue

if __name__ == '__main__':
    main()