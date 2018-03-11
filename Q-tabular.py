import socket
import re
import select
import numpy as np
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers


player = sys.argv[1]


class Config():
    epsilon_train = .1
    gamma = 0.9
    lr = 0.001
    num_players_per_team = 2
    num_actions = 9 + num_players_per_team - 1
    state_size = 2*(2*num_players_per_team) + 1
    num_rows = 6
    num_cols = 8


Q_table = np.zeros((Config.num_rows, Config.num_cols, Config.num_rows, Config.num_cols, 
    Config.num_rows, Config.num_cols, Config.num_rows, Config.num_cols, 2*Config.num_players_per_team, Config.num_actions), dtype=np.float32)


def get_best_action(state):
    """
    Returns best action according to the network

    Args:
        state: observation from gym
    Returns:
        tuple: action, q values
    """
    action_values = Q_table[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]]
    return np.argmax(action_values)


def get_action(state):
    """
    Returns action with some epsilon strategy

    Args:
        state: observation from gym
    """
    best_action = get_best_action(state)
    random_action = np.random.randint(0, Config.num_actions)
    prob = np.random.random()
   	return random_action if prob < Config.epsilon_train else best_action

# my row, my col, teammate row, teammate col, opp1 row, opp1 col, opp2 row, opp2 col, ball index
def get_state(agent, obs):
    state = np.zeros(Config.state_size, dtype=int)
    team_players_start_index = 2
    opp_players_start_index = 2*Config.num_players_per_team
    ball_loc = None
    for o in obs:
        if o[0] == "loc":
            state[0] = o[1][0]
            state[1] = o[1][1]
        elif o[0] == "player" and agent.uni_number != o[1][1]:
            if agent.left_team == o[1][0]:
                state[team_players_start_index] = o[1][2]
                state[team_players_start_index+1] = o[1][3]
                team_players_start_index += 2
            else:
                state[opp_players_start_index] = o[1][2]
                state[opp_players_start_index+1] = o[1][3]
                opp_players_start_index += 2
        elif o[0] == "ball":
            ball_loc = [o[1][0], o[1][1]]
    for i in range(2*Config.num_players_per_team):
        if ball_loc[0] == state[2*i] and ball_loc[1] == state[2*i + 1]:
            state[-1] = i
            break
    return state                    


def update_Q(state, action, reward, new_state):
    Q_old = Q_table[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], action]
    Q_opt_new_state = Q_table[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4], new_state[5], new_state[6], new_state[7], new_state[8]].max()
    Q_new = (1. - Config.lr)*Q_old + Config.lr*(reward + Config.gamma*Q_opt_new_state)
    Q_table[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], action] = Q_new

 
def train():
    """
    Performs training of Q

    Args:
        exp_schedule: Exploration instance s.t.
            exp_schedule.get_action(best_action) returns an action
        lr_schedule: Schedule for learning rate
    """

    # initialize replay buffer and variables
    score_team = 0
    score_opp = 0
    i = 0
    while True:
        i += 1
        # agents = []
        # agent_obs = []
        # for a in range(Config.num_players_per_team):
        agent = AgentInterface('SMART', player)
        agent.set_home(2+player, 2) 
        # agents.append(agent)
        obs = agent.observe_from_server()
        # agent_obs.append(obs)

        state_prev = None
        action_prev = None
        # for a in range(Config.num_players_per_team):
        while ("start", 0) not in obs:
            obs = agent.observe_from_server()
            state_prev = get_state(agent, obs)
        while True:
            # for a in range(Config.num_players_per_team):
            obs = agent.observe_from_server()
            new_cycle = False
            for o in obs:
                if o[0] == "cycle":
                    new_cycle = True
                    break
            if new_cycle:
                state_new = get_state(agent, obs)
                if action_prev is not None:
                    score = None
                    for o in obs:
                        if o[0] == "score":
                            if agent.left_team:
                                score = [o[1][0], o[1][1]]
                            else:
                                score = [o[1][1], o[1][0]]
                    reward_prev = 0.
                    if score[0] > score_team:
                        reward_prev = 1.
                    elif score[1] > score_opp:
                        reward_prev = -1.
                    else:
                        reward_prev = -0.1
                    score_team = score[0]
                    score_opp = score[1]
                    update_Q(state_prev, action_prev, reward_prev, state_new)
                action_new = get_action(state_new)
                if action_new <= 8:
                    agent.send_action("move", action_new)
                else:
                    teammate = 2 if player == 1 else 1
                    agent.send_action("pass", teammate)
                state_prev = state_new.copy()
                action_prev = action_new                
        

    


if __name__ == '__main__':
	train()