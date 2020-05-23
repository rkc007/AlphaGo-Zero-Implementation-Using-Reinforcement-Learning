import numpy as np
import sys
from goSim import GoEnv
sys.path.insert(1, './utils_6/')
from policyValueNet import PolicyValueNet
from policyValueNet import args as nnargs
from MCTS import MCTS
from copy import copy, deepcopy
import os
from config import *
from utils import *
from enums import Colour
import numpy  as np
import timeit
import time, random
from time_handler import deadline, TimedOutExc


class AlphaGoPlayer():
    def __init__(self, init_state, seed, player_color, board_size=13, timesteps=8):
        # print("PLAYER COLOR = ", player_color)
        self.init_state = init_state
        self.seed = seed
        self.player_color = player_color
        self.timesteps = timesteps
        if(self.player_color == Colour.BLACK.value):
            sim_color = 'black'
        else:
            sim_color = 'white'
        self.simulator = GoEnv(player_color='black', observation_type='image3c', illegal_move_mode="raise", board_size=board_size, komi=7.5)
        self.simulator.reset()
        self.policy_value_net = PolicyValueNet(nnargs)
        curr_max = -1
        for file in os.listdir('./model_6/'):
            if(file[0] != '.'):
                files = file.split('-')
                curr_max = max(curr_max, int(files[0]))
        self.policy_value_net.load_checkpoint(tag=str(curr_max), folder='model_6')
        self.mcts = MCTS(self.policy_value_net, copySimulator(self.simulator))
        self.currState = np.zeros((NUM_FEATURES, 13, 13))
        self.currState[-1, :, :] = self.player_color

    def sampleAction(self, policy):
    
        action = np.random.choice(NUM_ACTIONS, p=policy)
        assert (action >= 0 and action <= NUM_ACTIONS - 1), "Valid action not selected"

        return action

    # Simulator passes observation as current state
    @deadline(5)
    def get_action(self, cur_obs, opponent_action):
        start_t = time.time()
        # print("OPPONENT ACTION = ", opponent_action)
        if(opponent_action != -1):
            self.simulator.set_player_color(3 - self.player_color)
            _, _, _, _, _, _ = self.simulator.step(opponent_action)
        newState = np.array(self.currState)
        newState[2:-1, :, :]  = self.currState[0:NUM_FEATURES - 3, :, :]
        newState[-1, :, :] =   self.currState[-1, :, :]
        if(self.player_color == Colour.WHITE.value):
            newState[0, :, :] = cur_obs[1, :, :]
            newState[1, :, :] = cur_obs[0, :, :]
        else:
        # print("Colour Black") 
            newState[0:2, :, :] = cur_obs[0:2, :, :]

        self.simulator.set_player_color(self.player_color)
        self.currState = deepcopy(newState)
        self.mcts.updateSimulator(copySimulator(self.simulator))
        
        assert(self.simulator.state.color == self.simulator.player_color)
        assert(self.player_color == self.simulator.player_color)
        
        try:
            policy = self.mcts.getPolicy(deepcopy(self.currState), prev_action=opponent_action, temp=0.1)
            action = self.sampleAction(policy)
                        
        except TimedOutExc as e:
            print("took too long")
            action = goSim._pass_action(self.board_size)
        _, _, win_reward, _, _, curr_score= self.simulator.decide_winner()
        # self.simulator.render()
        # print("WIN REWARD OUTER = ", win_reward)
        # print("Current score OUTER = ", curr_score)

        if(opponent_action == PASS_ACTION):
            _, _, win_reward, _, _, curr_score= self.simulator.decide_winner()
            # self.simulator.render()
            # print("WIN REWARD = ", win_reward)
            # print("Current score = ", curr_score)
            if(win_reward > 0):
                # self.simulator = GoEnv(player_color='black', observation_type='image3c', illegal_move_mode="raise", board_size=13, komi=7.5)
                # self.simulator.reset()
                action = PASS_ACTION
        print("ACTION CHOSEN = ", action)
        self.simulator.step(action)
        end_t = time.time()
        print("Total time taken for action = ", end_t - start_t)
        return action


    