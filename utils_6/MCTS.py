import sys
sys.path.insert(1, './')
import numpy as np
from policyValueNet import PolicyValueNet
from policyValueNet import args as neuralNetArgs
from enums import Colour
from config import *
from utils import obsToString, stateToObs, obsToState, invertObs, copySimulator, getNextState
from copy import copy, deepcopy
import math
import time

class MCTS():

    def __init__(self, nNet, simulator):
        self.nNet  = nNet
        self.Qsa = {}
        self.Psa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Wsa = {}
        self.Rs = {} # game ended -> 0, win -> 1, loss -> -1 (for player 1)
        self.valid_moves = {}
        self.num_simulations = NUM_SIMULATIONS
        self.simulator = simulator
        self.simulations_time = SIMULATIONS_TIME
        # print(simulator.player_color)
        # print(simulator.state.color)
        self.currSimulator = copySimulator(simulator)
        self.policyTime = 0.0

    def updateSimulator(self, simulator):
        self.simulator =  copySimulator(simulator)
        self.currSimulator = copySimulator(simulator)
        # print(num_simla)
        # self.simulator.state.color = Colour.BLACK.value
        # self.simulator.player_color = Colour.BLACK.value


    # run MCTS with player 1 (BLACK)
    # invert it when the player is WHITE
    def getPolicy(self, state, prev_action=None, temp=1):
        start_t = time.time()
        # from IPython import embed; embed()
        # print("Inside MCTS colour ", state[16 ,0, 0])
        # if(state[16,0,0] == Colour.WHITE.value):
        #     state[16, :, :] = Colour.BLACK.value
        player_color = self.simulator.player_color
        
        sims = 0
        while(True):
            # print("Current player to play = ", player_color)
            # print("Simulation no. = ", i)
            
            # print('------------------NEW SIMULATION-------------------')
            # print(state[16 ,0, 0])
            try:
                self.search(deepcopy(state), 0, prev_action == PASS_ACTION)
                sims += 1
                if(time.time() - start_t > self.simulations_time):
                    # print("total simulations done = ", sims)
                    break
            except:
                break
        

        curObs = stateToObs(state)
        strObs = obsToString(curObs)
        counts = np.zeros(NUM_ACTIONS)

        for a in range(NUM_ACTIONS):
            if ((strObs, player_color, prev_action == PASS_ACTION), a) in self.Nsa:
                counts[a] = self.Nsa[((strObs, player_color, prev_action == PASS_ACTION), a)]
                if(counts[a] > 0):
                    # print(a, counts[a])
                    # self.simulator.render()
                    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                    assert(self.simulator.is_legal_action(a))


        if(temp == 0):
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        
        counts = [x ** (1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        # print("Policy = ", probs)
        end_t = time.time()
        totalTime = end_t - start_t
        # print("Policy time = ", self.policyTime)
        # print("Total time = ", totalTime)
        self.policyTime = 0.0
        return probs

    def getValidMoves(self):
        assert(self.currSimulator is not None)
        validActions = np.zeros(NUM_ACTIONS, dtype='int32')
        toPass = True
        # print("Current simulator colour = ", self.currSimulator.player_color)

        for action in range(NUM_ACTIONS):
            # if(self.currSimulator is not None):
            is_legal_action = self.currSimulator.is_legal_action(action)
            # else:
            #     is_legal_action = self.simulator.is_legal_action(action)
            # # print("Checking action = ", action)
            # print(is_legal_action)
            if(is_legal_action):
                validActions[action] = 1
                toPass = False
                # print("action = ", action)
        if toPass:
            validActions[169] = 1
        else:
            validActions[169] = 0
        validActions[170] = 0
        # print("Total valid actions = ", np.sum(validActions))        
        if(np.sum(validActions) == 0):
            # print("No valid action")
            # print(validActions)
            validActions[169] = 1

        assert(np.sum(validActions) > 0)
        return validActions

    def getNextObs(self, action):
        assert (self.currSimulator) is not None
        # self.currSimulator = copySimulator(self.simulator)
        # print("-------------------getNextState-----------------")
        # print('Initial -> Action: {}, Player: {}, State: {}'.format(
        #     action,
        #     Colour(self.currSimulator.player_color).__str__(),
        #     Colour(self.currSimulator.state.color).__str__()
        #     ))

        obs_t = None
        r_t = None
        assert (self.currSimulator.state.color == self.currSimulator.player_color), "State color: {}, Playe Color: {}".format(self.currSimulator.state.color, self.currSimulator.player_color)
        # if not self.currSimulator.is_legal_action(action):
            # self.currSimulator.render()
            # print(action)
        assert(self.currSimulator.is_legal_action(action))
        obs_t, action, r_t, done, info, cur_score = self.currSimulator.step(action)
        self.currSimulator.set_player_color(3 - self.currSimulator.player_color)
        return obs_t, r_t


        # if(self.currSimulator.state.color == Colour.WHITE.value):
        #     # self.currSimulator.state.color = Colour.BLACK.value
        #     # self.currSimulator.player_color = Colour.BLACK.value
        #     obs_t, action, r_t, done, info, cur_score = self.currSimulator.step(action)
        #     # self.currSimulator.state.color = Colour.BLACK.value
        #     # print(r_t, done)
        #     # print('Observation Inverted.') # Invert when the state of the simulator is WHITE
        #     obs_t = invertObs(obs_t)
        # else:
        #     # print('Observation Not Inverted.')

        #     obs_t, action, r_t, done, info, cur_score = self.currSimulator.step(action)
        #     # print(r_t, done)
        #     self.currSimulator.state.color = Colour.WHITE.value
        #     self.currSimulator.player_color = Colour.WHITE.value
        # # print('Final -> Player: {}, State: {}'.format(
        # #     Colour(self.currSimulator.player_color).__str__(),
        # #     Colour(self.currSimulator.state.color).__str__()
        # # ))
        # # print(obsToString(obs_t), r_t)
        # return obs_t, r_t

    # State is of shape (17 x 13 x 13) -- STATE WITH RESPECT TO BLACK
    def search(self, state, reward, pass_action):
        # print(state[16, 0, 0])
        # print(Colour.BLACK.value)

        # assert(state[16, 0, 0] == Colour.BLACK.value)
        # player_colour = Colour.BLACK.value
        # Get Obs from State (Picked Top 2)
        obs = stateToObs(state)
        # obs = np.zeros((3, state.shape[1], state.shape[2]))
        # obs[0, :, :] = state[0, :, :]
        # obs[1, :, :] = state[1, :, :]
        # obs[2, :, :] = np.logical_not(np.logical_or(obs[0, :, :], obs[1, :, :]))

        # Convert it into String representation
        strState = obsToString(obs)
        player_color = self.currSimulator.player_color
        # print("Actual player = ", player_color)
        # print(strState)
        if (strState, player_color, pass_action) not in self.Rs:
            # print("Reward first time")
            self.Rs[(strState, player_color, pass_action)] = reward
        else:
            # Correct code for action = 169, 170
            assert (reward == self.Rs[(strState, player_color, pass_action)])
                # print(strState)
                # print("Next reward: {}, dict reward: {}".format(reward, self.Rs[(strState, player_color, pass_action)]))
                # print(self.simulator.player_color)
                # print("CORRRECCTTTTT IT IN FUTURE!!!!!!!!!!!!!!!!!!!")
                # if(reward != 0):
                #     self.Rs[(strState, player_color, pass_action)] = reward

        if(self.Rs[(strState, player_color, pass_action)] != 0):
            # terminal node
            self.currSimulator = copySimulator(self.simulator)
            return -self.Rs[(strState, player_color, pass_action)]
        
        if (strState, player_color, pass_action) not in self.Psa:
            # print("State first time estimating pp")

            start_t = time.time()
            ps, vs = self.nNet.predict(state)
            end_t = time.time()
            self.policyTime += end_t - start_t
            # print('Time elapsed for prediction = {}'.format(
                        # end_t - start_t
                    # ))            
            assert(player_color == self.currSimulator.player_color)
            valids = self.getValidMoves()
            # print(valids)
            ps = ps * valids
            self.Psa[(strState, player_color, pass_action)] = ps
            assert(np.sum(ps) > 0)
            if(np.sum(ps) > 0):
                self.Psa[(strState, player_color, pass_action)] /= np.sum(ps)
            # else:
            #     # Assigning equal probability to all valid moves
            #     self.Psa[strState] += valids
            #     self.Psa[strState] /= np.sum(ps)
            self.valid_moves[(strState, player_color, pass_action)] = valids
            self.Ns[(strState, player_color, pass_action)] = 0
            self.currSimulator = copySimulator(self.simulator)
            return -vs

        assert((strState, player_color, pass_action) in self.valid_moves)
        # print("State discovered before")

        valids = self.valid_moves[(strState, player_color, pass_action)]
        cur_best = -float('inf')
        best_act = -1

        for a in range(NUM_ACTIONS):
            if valids[a] and self.currSimulator.is_legal_action(a):
               # print(self.Psa[strState].shape)
                # print(self.Ns[strState].shape)

                if ((strState, player_color, pass_action),a) in self.Qsa:
                    u = self.Qsa[((strState, player_color, pass_action),a)] + CPUCT*self.Psa[(strState, player_color, pass_action)][a]*math.sqrt(self.Ns[(strState, player_color, pass_action)])/(1+self.Nsa[((strState, player_color, pass_action),a)])
                else:
                    u = CPUCT*self.Psa[(strState, player_color, pass_action)][a]*math.sqrt(self.Ns[(strState, player_color, pass_action)] + EPSILON)     # Q = 0 ?

                # print(u.shape)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        assert(best_act != -1)
        # if(a == PASS_ACTION):
        #     passes += 1
        next_obs, next_reward = self.getNextObs(a)

        # Invert Obs to get Observation with respect to Black
        # print(player_colour)
        next_state = getNextState(state, next_obs) # Color is Black

        # print("Current player:{}, action taken:{}".format(player_color, a))
        v = self.search(next_state, -next_reward, (a == PASS_ACTION))

        if ((strState, player_color, pass_action),a) in self.Qsa:
            self.Wsa[((strState, player_color, pass_action), a)] += v
            self.Nsa[((strState, player_color, pass_action), a)] += 1
            self.Qsa[((strState, player_color, pass_action), a)] = self.Wsa[((strState, player_color, pass_action), a)] / self.Nsa[((strState, player_color, pass_action), a)]
        else:
            self.Wsa[((strState, player_color, pass_action), a)] = v
            self.Qsa[((strState, player_color, pass_action), a)] = v
            self.Nsa[((strState, player_color, pass_action), a)] = 1

        self.Ns[(strState, player_color, pass_action)] += 1
        self.currSimulator = copySimulator(self.simulator)

        return -v


