import socket
import re
import select
import numpy as np
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Config():
    epsilon_train = 0.05
    gamma = 1.0
    lr = 0.001
    num_players_per_team = 2
    num_actions = 10

class QN:
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self):
        self.build()

    def add_placeholders_op(self):
        state_size = 2*Config.num_players_per_team + 1
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
        random_actions = np.random.randint(0, Config.num_actions, states.shape[0])
        probs = np.random.random(states.shape[0])
       	return np.where(probs < Config.epsilon_train, random_actions, best_actions)

 
    def train(self):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        i = 0
        while True:
        	i += 1
            smart_agents = []
            for a in range(Config.num_players_per_team):
                smart_agents.append(AgentInterface(team_name='SMART', a))
	        self.smart_agent1 = Intersection()
	        state = self.inters.states
	        new_state = None
	        loss_sum = 0.
	        j = 0
	        while not self.inters.is_end():
	        	actions = self.get_action(state)
	        	rewards, new_state = self.inters.apply_action(actions)
	        	loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.s: state, self.r:  rewards, self.a: actions})
	        	j += 1
	        	loss_sum += loss
	        	new_state = state
	       	loss_avg = loss_sum / j
	        print i, self.inters.num_timesteps, self.inters.num_completed, self.inters.tot_reward, loss_avg    


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
