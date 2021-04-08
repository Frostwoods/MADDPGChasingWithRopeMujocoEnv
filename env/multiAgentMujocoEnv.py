import numpy as np
import math
import mujoco_py as mujoco

getPosFromAgentState = lambda state: np.array([state[0], state[1]])

getVelFromAgentState = lambda agentState: np.array([agentState[2], agentState[3]])


class ResetUniformWithoutXPosForLeashed:
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

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        tiedBasePos = qPos[self.numJointEachSite * self.tiedBasePosAgentIndex: self.numJointEachSite * (self.tiedBasePosAgentIndex + 1)]
        sampledRopeLength = np.random.uniform(low = 0, high = self.numRopePart * self.maxRopePartLength)
        sampledPartLength = np.arange(sampledRopeLength/(self.numRopePart + 1), sampledRopeLength, sampledRopeLength/(self.numRopePart + 1))[:self.numRopePart]
        theta = np.random.uniform(low = 0, high = math.pi)

        tiedFollowPosAgentPos = tiedBasePos + np.array([sampledRopeLength * np.cos(theta), sampledRopeLength * np.sin(theta)])
        qPos[self.numJointEachSite * self.tiedFollowPosAgentIndex : self.numJointEachSite * (self.tiedFollowPosAgentIndex + 1)] = tiedFollowPosAgentPos
        ropePartPos = np.array(list(zip(sampledPartLength * np.cos(theta), sampledPartLength * np.sin(theta)))) + tiedBasePos
        qPos[-self.numJointEachSite * self.numRopePart : ] = np.concatenate(ropePartPos)

        qVelSampled = np.concatenate([np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel - self.numRopePart * self.numJointEachSite),\
                                      np.zeros(self.numRopePart * self.numJointEachSite)])
        qVel = self.qVelInit + qVelSampled

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


class TransitionFunctionWithoutXPos:
    def __init__(self, simulation,numSimulationFrames,visualize, isTerminal, reshapeActionList):
        self.simulation = simulation
        self.isTerminal = isTerminal
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)
        self.reshapeActionList=reshapeActionList
        self.visualize=visualize
        if visualize:
            self.physicsViewer = mujoco.MjViewer(simulation)

    def __call__(self, state, actions):
        actions = [reshapeAction(action) for action,reshapeAction in zip(actions,self.reshapeActionList)]
        state = np.asarray(state)
        actions = np.asarray(actions)
        numAgent = len(state)
        oldQPos = state[:, 0:self.numJointEachSite].flatten()
        oldQVel = state[:, -self.numJointEachSite:].flatten()
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()
        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()
            if self.visualize:
                self.physicsViewer.render()
            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel

            agentNewQPos = lambda agentIndex: newQPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
            agentNewQVel = lambda agentIndex: newQVel[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
            agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewQVel(agentIndex)])
            newState = np.asarray([agentNewState(agentIndex) for agentIndex in range(numAgent)])

            if self.isTerminal(newState):
                break

        return newState


class IsCollision:
    def __init__(self, getPosFromState, killZone = 0):
        self.getPosFromState = getPosFromState
        self.killZone = killZone

    def __call__(self, agent1State, agent2State, agent1Size, agent2Size):
        posDiff = self.getPosFromState(agent1State) - self.getPosFromState(agent2State)
        dist = np.sqrt(np.sum(np.square(posDiff)))
        minDist = agent1Size + agent2Size + self.killZone
        return True if dist < minDist else False


class RewardWolf:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward=10):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.collisionReward = collisionReward

    def __call__(self, state, action, nextState):
        wolfReward = 0

        for wolfID in self.wolvesID:
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]
            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    wolfReward += self.collisionReward
        reward = [wolfReward] * len(self.wolvesID)
        # print('wolfreward ', wolfReward)
        return reward


class PunishForOutOfBound:
    def __init__(self):
        self.physicsDim = 2

    def __call__(self, agentPos):
        punishment = 0
        for i in range(self.physicsDim):
            x = abs(agentPos[i])
            punishment += self.bound(x)
        return punishment

    def bound(self, x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)


class RewardSheep:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, getPosFromState, isCollision, punishForOutOfBound,
                 collisionPunishment=10):
        self.wolvesID = wolvesID
        self.getPosFromState = getPosFromState
        self.entitiesSizeList = entitiesSizeList
        self.sheepsID = sheepsID
        self.isCollision = isCollision
        self.collisionPunishment = collisionPunishment
        self.punishForOutOfBound = punishForOutOfBound

    def __call__(self, state, action, nextState):  # state, action not used
        reward = []
        for sheepID in self.sheepsID:
            sheepReward = 0
            sheepNextState = nextState[sheepID]
            sheepNextPos = self.getPosFromState(sheepNextState)
            sheepSize = self.entitiesSizeList[sheepID]

            sheepReward -= self.punishForOutOfBound(sheepNextPos)
            for wolfID in self.wolvesID:
                wolfSize = self.entitiesSizeList[wolfID]
                wolfNextState = nextState[wolfID]
                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    sheepReward -= self.collisionPunishment
            reward.append(sheepReward)

        return reward


class ResetFixWithoutXPos:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, numBlock):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.numBlock = numBlock
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)

    def __call__(self, fixPos, fixVel, blocksState):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.qPosInit + np.array(fixPos)
        qVel = self.qVelInit + np.array(fixVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getAgentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        agentState = [getAgentState(agentIndex) for agentIndex in range(self.numAgent)]

        startState = np.asarray(agentState + blocksState)

        return startState


class IsOverlap:
    def __init__(self, minDistance):
        self.minDistance = minDistance

    def __call__(self, blocksState, proposalState):
        for blockState in blocksState:
            distance = np.linalg.norm(np.array(proposalState[:2]) - np.array(blockState[:2]))
            if distance < self.minDistance:
                return True
        return False


class SampleBlockState:
    def __init__(self, numBlocks, getBlockPos, getBlockSpeed, isOverlap):
        self.numBlocks = numBlocks
        self.getBlockPos = getBlockPos
        self.getBlockSpeed = getBlockSpeed
        self.isOverlap = isOverlap

    def __call__(self):
        blocksState = []
        for blockID in range(self.numBlocks):
            proposalState = list(self.getBlockPos()) + list(self.getBlockSpeed())
            while self.isOverlap(blocksState, proposalState):
                proposalState = list(self.getBlockPos()) + list(self.getBlockSpeed())
            blocksState.append(proposalState)

        return blocksState


class Observe:
    def __init__(self, agentID, wolvesID, sheepsID, blocksID, getPosFromState, getVelFromAgentState):
        self.agentID = agentID
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.blocksID = blocksID
        self.getEntityPos = lambda state, entityID: getPosFromState(state[entityID])
        self.getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])

    def __call__(self, state):
        blocksPos = [self.getEntityPos(state, blockID) for blockID in self.blocksID]
        agentPos = self.getEntityPos(state, self.agentID)
        blocksInfo = [blockPos - agentPos for blockPos in blocksPos]

        posInfo = []
        for wolfID in self.wolvesID:
            if wolfID == self.agentID: continue
            wolfPos = self.getEntityPos(state, wolfID)
            posInfo.append(wolfPos - agentPos)

        velInfo = []
        for sheepID in self.sheepsID:
            if sheepID == self.agentID: continue
            sheepPos = self.getEntityPos(state, sheepID)
            posInfo.append(sheepPos - agentPos)
            sheepVel = self.getEntityVel(state, sheepID)
            velInfo.append(sheepVel)

        agentVel = self.getEntityVel(state, self.agentID)
        return np.concatenate([agentVel] + [agentPos] + blocksInfo + posInfo + velInfo)

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

class ResetUniformWithoutXPos:
    def __init__(self, simulation, numAgent, numBlock, sampleAgentsQPos, sampleAgentsQVel, sampleBlockState):
        self.simulation = simulation
        self.numAgent = self.simulation.model.nsite
        self.numBlock = numBlock
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.sampleAgentsQPos = sampleAgentsQPos
        self.sampleAgentsQVel = sampleAgentsQVel
        self.sampleBlockState = sampleBlockState

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.sampleAgentsQPos()
        qVel = self.sampleAgentsQVel()

        blocksState = self.sampleBlockState()

        for block in range(
                self.numBlock):  # change blocks pos in mujoco simulation,see xml for more details,[floor+agents+blocks] [x,y,z]
            self.simulation.model.body_pos[2 + self.numAgent + block][:2] = blocksState[block][:2]
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getAgentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        agentState = [getAgentState(agentIndex) for agentIndex in range(self.numAgent)]
        startState = np.asarray(agentState + blocksState)

        return startState


# class ResetUniformWithoutXPos:
#     def __init__(self, simulation, qPosInit, qVelInit, numAgent,numBlock, qPosInitNoise, qVelInitNoise,sampleBlockState):
#         self.simulation = simulation
#         self.qPosInit = np.asarray(qPosInit)
#         self.qVelInit = np.asarray(qVelInit)
#         self.numAgent = self.simulation.model.nsite
#         self.numBlock = numBlock
#         self.qPosInitNoise = qPosInitNoise
#         self.qVelInitNoise = qVelInitNoise
#         self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)

#         self.sampleBlockState = sampleBlockState
#     def __call__(self):
#         numQPos = len(self.simulation.data.qpos)
#         numQVel = len(self.simulation.data.qvel)

#         qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
#         qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

#         blocksState = self.sampleBlockState()

#         for block in range(self.numBlock):#change blocks pos in mujoco simulation
#             self.simulation.model.body_pos[2+self.numAgent+block][:2]=blocksState[block][:2]
#         self.simulation.data.qpos[:] = qPos
#         self.simulation.data.qvel[:] = qVel
#         self.simulation.forward()

#         agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
#         agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
#         getAgentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
#         agentState = [getAgentState(agentIndex) for agentIndex in range(self.numAgent)]
#         startState=np.asarray(agentState+blocksState)


#         return startState





class TransitionFunction:
    def __init__(self, simulation, numAgents, numSimulationFrames, visualize, isTerminal, reshapeAction):
        self.simulation = simulation
        self.numAgents = numAgents
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.visualize = visualize
        self.isTerminal = isTerminal
        self.reshapeAction = reshapeAction
        if visualize:
            self.physicsViewer = mujoco.MjViewer(simulation)

    def __call__(self, state, actions):
        actions = [self.reshapeAction(action) for action in actions]
        state = np.asarray(state)
        actions = np.asarray(actions)
        oldQPos = np.array(
            [QPos for agent in state[:self.numAgents] for QPos in agent[:self.numJointEachSite]]).flatten()
        oldQVel = np.array(
            [QVel for agent in state[:self.numAgents] for QVel in agent[-self.numJointEachSite:]]).flatten()
        blocksState = [np.asarray(block) for block in state[self.numAgents:]]

        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()

        agentNewQPos = lambda agentIndex: newQPos[
                                          self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentNewQVel = lambda agentIndex: newQVel[
                                          self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getSpeed = lambda Vel: np.linalg.norm(Vel)
        agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewQVel(agentIndex)])

        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()
            if self.visualize:
                self.physicsViewer.render()
            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel

            newState = [agentNewState(agentIndex) for agentIndex in range(self.numAgents)]
            if self.isTerminal(newState):
                break

        newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
        newState = [agentNewState(agentIndex) for agentIndex in range(self.numAgents)]
        newState = np.asarray(newState + blocksState)
        return newState


class IsTerminal:
    def __init__(self, minXDis, getAgent0Pos, getAgent1Pos):
        self.minXDis = minXDis
        self.getAgent0Pos = getAgent0Pos
        self.getAgent1Pos = getAgent1Pos

    def __call__(self, state):
        state = np.asarray(state)
        pos0 = self.getAgent0Pos(state)
        pos1 = self.getAgent1Pos(state)
        L2Normdistance = np.linalg.norm((pos0 - pos1), ord=2)
        terminal = (L2Normdistance <= self.minXDis)

        return terminal
