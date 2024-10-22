import socket
import re
import select
import numpy as np
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers


class Config():
    epsilon_train = .15
    gamma = 0.8
    lr = 0.001
    num_players_per_team = 2
    num_actions = 9 + num_players_per_team - 1
    state_size = 2*(2*num_players_per_team + 1)

class QN:
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self):
        self.build()

    def add_placeholders_op(self):
        state_size = Config.state_size
        self.s = tf.placeholder(tf.float32, (None, state_size))
        self.a = tf.placeholder(tf.int32, (None))
        self.r = tf.placeholder(tf.float32, (None))

    def linear_get_q_values_op(self, state, scope, reuse=False):
        num_actions = Config.num_actions
        out = state
        print out
        out = layers.flatten(out, scope=scope)
        print out
        out = layers.fully_connected(out, num_actions, activation_fn=None, reuse=reuse, scope=scope)
        print out
        ##############################################################
        ######################## END YOUR CODE #######################

        return out

    def deep_get_q_values_op(self, state, scope, reuse=False):
    	num_actions = Config.num_actions
    	out = state
        with tf.variable_scope(scope, reuse=reuse) as _:
            out = layers.fully_connected(out, 512)
            out = layers.fully_connected(out, num_actions, None) 
        ##############################################################
        ######################## END YOUR CODE #######################

        return out

    def add_loss_op(self, q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = Config.num_actions
        upd = self.r + Config.gamma * tf.reduce_max(q, axis=1) - tf.reduce_sum(tf.one_hot(self.a, num_actions)*q, axis=1)
        self.loss = tf.reduce_mean(upd**2.)

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):

        optimizer = tf.train.AdamOptimizer(Config.lr)
        self.train_op = optimizer.minimize(self.loss)
        
        ##############################################################
        ######################## END YOUR CODE #######################


    def build(self):
        """
        Build model
        """
        
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        self.q = self.deep_get_q_values_op(self.s, scope="q", reuse=False)

        # add square loss
        self.add_loss_op(self.q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")


    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass


    def initialize(self):
        """
        Initialize variables if necessary
        """
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # # tensorboard stuff
        # self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def get_best_action(self, state):
        """
        Returns best action according to the network
    
        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: state})
        return np.argmax(action_values, axis=1), action_values


    def get_action(self, states):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        best_actions = self.get_best_action(states)[0]
        random_actions = np.random.randint(0, Config.num_actions, len(states))
        probs = np.random.random(len(states))
       	return np.where(probs < Config.epsilon_train, random_actions, best_actions)

    def get_state(self, agent, obs):
        state = np.zeros(Config.state_size)
        team_players_start_index = 2
        opp_players_start_index = 2*Config.num_players_per_team
        for o in obs:
            if o[0] == "loc":
                state[0] = o[1][0]
                state[1] = o[1][1]
            elif o[0] == "player":
                if agent.left_team == o[1][0] and agent.uni_number != o[1][1]:
                    state[team_players_start_index] = o[1][2]
                    state[team_players_start_index+1] = o[1][3]
                    team_players_start_index += 2
                elif agent.left_team != o[1][0]:
                    state[opp_players_start_index] = o[1][2]
                    state[opp_players_start_index+1] = o[1][3]
                    opp_players_start_index += 2
            elif o[0] == "ball":
                state[-2] = o[1][0]
                state[-1] = o[1][1]
        return state                    




 
    def train(self):
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
            agents = []
            agent_obs = []
            for a in range(Config.num_players_per_team):
                agent = AgentInterface('SMART', a)
                agent.set_home(3+a, 2)
                agents.append(agent)
                obs = agent.observe_from_server()
                agent_obs.append(obs)

            states = np.zeros((Config.num_players_per_team, Config.state_size))
            for a in range(Config.num_players_per_team):
                while ("start", 0) not in agent_obs[a]:
                    agent_obs[a] = agents[a].observe_from_server()
                    states[a] = self.get_state(agents[a], agent_obs[a])
            new_states = np.zeros((Config.num_players_per_team, Config.state_size))
            while True:
                for a in range(Config.num_players_per_team):
                    agent_obs[a] = agents[a].observe_from_server()
                    new_cycle = False
                    for o in agent_obs[a]:
                        if o[0] == "cycle":
                            new_cycle = True
                            break
                    if new_cycle:
                        action = self.get_action([states[a]])[0]
                        if action <= 8:
                            agents[a].send_action("move", action)
                        else:
                            if action - 9 < a:
                                agents[a].send_action("pass", action-9)
                            else:
                                agents[a].send_action("pass", action-8)
                        new_states[a] = self.get_state(agents[a], agent_obs[a])
                        score = None
                        for o in agent_obs[a]:
                            if o[0] == "score":
                                if agents[a].left_team:
                                    score = [o[1][0], o[1][1]]
                                else:
                                    score = [o[1][1], o[1][0]]
                        reward = 0.
                        if score[0] > score_team:
                            reward = 1.
                        elif score[1] > score_opp:
                            reward = -1.
                        score_team = score[0]
                        score_opp = score[1]
                        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.s: [states[a]], self.r:  [reward], self.a: [action]})
                        states[a] = new_states[a]




    def run(self):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # model
        self.train()
        

    



if __name__ == '__main__':
	model = QN()
	model.run()
