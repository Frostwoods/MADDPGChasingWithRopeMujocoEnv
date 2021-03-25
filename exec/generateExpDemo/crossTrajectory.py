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

        maxEpisode = 200000
        evaluateEpisode = 200000
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
    modelSaveName = '2expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
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
    expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectory', modelSaveName,'normal')
    # if not os.path.exists(expTrajectoriesSaveDirectory):
    #     os.makedirs(expTrajectoriesSaveDirectory)
    trajectorySaveExtension = '.pickle'
    fixedParameters = {'damping': damping,'frictionloss':frictionloss,'masterForce':masterForce,'evalNum':evalNum,'evaluateEpisode':evaluateEpisode}
    generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    expTrajectorySavePath = generateExpTrajectorySavePath({})
    print(expTrajectorySavePath)
    originalTrajList=loadFromPickle(expTrajectorySavePath)
    newTrajList=[]
    newTrajListForDraw=[]
    for trajIndex,traj in enumerate(originalTrajList):
        print(trajIndex)
        crossIndex=random.randint(0,evalNum-1)
        while crossIndex == trajIndex:
            crossIndex=random.randint(0,evalNum-1)
        print(traj[0])
        newTraj = [[replaceState[0],originalState[1],originalState[2],originalState[3]] for originalState,replaceState in zip (traj,originalTrajList[crossIndex])]
        newTrajForDraw = [[[replaceState[0],originalState[1],originalState[2],originalState[3]]] for originalState,replaceState in zip (traj,originalTrajList[crossIndex])]
        newTrajListForDraw.append(newTrajForDraw)
        newTrajList.append(newTraj)

    expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'expTrajectory', modelSaveName,'Cross')
    if not os.path.exists(expTrajectoriesSaveDirectory):
        os.makedirs(expTrajectoriesSaveDirectory)
    generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    expTrajectorySavePath = generateExpTrajectorySavePath({})
    saveToPickle(newTrajList, expTrajectorySavePath)

    if visualizeTraj:
        trajSaveName = '2expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed'
        pictureFolder = os.path.join(dataFolder, 'demo', trajSaveName,'Cross','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))



        if not os.path.exists(pictureFolder):
            os.makedirs(pictureFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [masterColor] * numDistractor
        render = Render(entitiesSizeList, entitiesColorList, numAgent,pictureFolder,saveImage, getPosFromAgentState)
        trajToRender = np.concatenate(newTrajListForDraw)
        render(trajToRender)

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.5]#[0.0, 1.0]
    manipulatedVariables['frictionloss'] =[0.1]# [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce']=[0.5]#[0.0, 2.0]
    # manipulatedVariables['distractorNoise']=[0,1,2,3,4]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        print(condition)
        generateSingleCondition(condition)
        # try:
            # generateSingleCondition(condition)
        # except:
        #     continue

if __name__ == '__main__':
    main()