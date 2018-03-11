import socket
import re
import select
import numpy as np
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys


player = int(sys.argv[1])


class Config():
    epsilon_train = .1
    gamma = 0.9
    lr = 0.1
    num_players_per_team = 2
    num_actions = 9 + num_players_per_team - 1
    state_size = 2*(2*num_players_per_team) + 1
    num_rows = 6
    num_cols = 8
    RewardEveryMovment = -2.
    RewardSuccessfulPass = -1.
    RewardHold = -1.
    RewardIllegalMovment = -3.
    RewardTeamCatchBall = 10.
    RewardTeamLooseBall = -10.
    RewardSelfCatchBall = 10.
    RewardSelfLooseBall = -10.
    RewardTeamScoreGoal = 10.
    RewardSelfScoreGoal = 10.
    RewardTeamRecvGoal = -10.
    RewardTeamOwnGoal = -15.
    RewardSelfOwnGoal = -20.
    RewardOpponentOwnGoal = 1.
    collaborative_rewards = True


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
            state[0] = o[1][0] - 1
            state[1] = o[1][1] - 1
        elif o[0] == "player" and agent.uni_number != o[1][1]:
            if agent.left_team == o[1][0]:
                state[team_players_start_index] = o[1][2] - 1
                state[team_players_start_index+1] = o[1][3] - 1
                team_players_start_index += 2
            else:
                state[opp_players_start_index] = o[1][2] - 1
                state[opp_players_start_index+1] = o[1][3] - 1
                opp_players_start_index += 2
        elif o[0] == "ball":
            ball_loc = [o[1][0] - 1, o[1][1] - 1]
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


def reward(state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new):
    preAmIBallOwner = state_prev[-1] == 0
    preAreWeBallOwner = state_prev[-1] < Config.num_players_per_team
    curAmIBallOwner = state_new[-1] == 0
    curAreWeBallOwner = state_new[-1] < Config.num_players_per_team
    isCollaborative = Config.collaborative_rewards

    # if we scored a goal
    if score_team_new > score_team_prev:
        if preAmIBallOwner:
            return Config.RewardSelfScoreGoal
        elif preAreWeBallOwner:
            if isCollaborative:
                return Config.RewardTeamScoreGoal
        else:
            return Config.RewardOpponentOwnGoal

    # if we received a goal
    if score_opp_new > score_opp_prev:
        if preAmIBallOwner:
            return Config.RewardSelfOwnGoal
        elif preAreWeBallOwner:
            return Config.RewardTeamOwnGoal
        else:
            return Config.RewardTeamRecvGoal

    if curAmIBallOwner and (not preAreWeBallOwner):
        return Config.RewardSelfCatchBall
    if curAreWeBallOwner and (not preAreWeBallOwner) and isCollaborative:
        return Config.RewardTeamCatchBall
    if (not curAreWeBallOwner) and preAmIBallOwner:
        return Config.RewardSelfLooseBall
    if (not curAreWeBallOwner) and preAreWeBallOwner and isCollaborative:
        return Config.RewardTeamLooseBall

    if action_prev == 9 and preAmIBallOwner and (not curAmIBallOwner) and curAreWeBallOwner:
        return Config.RewardSuccessfulPass

    if action_prev == 0:
        return Config.RewardHold

    if action_prev != 0 and action_prev != 9 and state_prev[0] == state_new[0] and state_prev[1] == state_new[1]:
        return Config.RewardIllegalMovment

    return Config.RewardEveryMovment
    
 
def train():
    """
    Performs training of Q

    Args:
        exp_schedule: Exploration instance s.t.
            exp_schedule.get_action(best_action) returns an action
        lr_schedule: Schedule for learning rate
    """

    # initialize replay buffer and variables
    score_team_prev = 0
    score_opp_prev = 0
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
                    score_team_new, score_opp_new = score[0], score[1]
                    reward_prev = reward(state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new)
                    print(reward_prev)
                    score_team_prev = score_team_new
                    score_opp_prev = score_opp_new
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