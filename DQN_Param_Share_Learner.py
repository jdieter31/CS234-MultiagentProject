
# coding: utf-8

# In[1]:


'''
IMPORTANT: PLAYERS MUST BE NUMBERED 0, 1, 2, ...

'''
# In[1]:

import socket
import re
import select
import sys
import time
import os
import numpy as np
import pickle
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers
from multiprocessing.dummy import Pool as ThreadPool 


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

# In[2]:

class Config():
    rows = 10
    columns = 18
    epsilon_train_start = 0.5
    epsilon_train_end = 0.05
    epsilon_decay_steps = 10000
    epsilon_soft = 0.05
    gamma = 0.9
    lr = 0.001
    num_players_per_team = 3
    num_actions = 9 + num_players_per_team - 1
    state_size = 4
    
    episode_len = 50
    # target_update_freq = episode_len
    
    # RewardLooseMatch = -10.
    # RewardWinMatch = 10.
    target_update_freq = 10
    sars_path = './param_sharing'
    if not os.path.exists(sars_path):
        os.makedirs(sars_path)
    
    RewardEveryMovment = 0.#-2.
    RewardSuccessfulPass = 0.#-1.
    RewardHold = 0.#-1.
    RewardIllegalMovment = 0.#-3.
    RewardTeamCatchBall = 0.#10.
    RewardTeamLooseBall = 0.#-10.
    RewardSelfCatchBall = 0.#10.
    RewardSelfLooseBall = 0.#-10.
    RewardTeamScoreGoal = 1.#50.
    RewardSelfScoreGoal = 1.#100.
    RewardTeamRecvGoal = -1#-100.
    RewardTeamOwnGoal = -1#-50.
    RewardSelfOwnGoal = -1#-100.
    RewardOpponentOwnGoal = 1.#1.
    collaborative_rewards = True
    

# In[3]:



class QNParamSharingLeaner():
    player = 0
    q_t=None
    t = 0
    episode_team_goals=0
    episode_opp_goals=0
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self,player=0,rows=10,columns=18):
        self.player = player
        Config.rows=rows
        Config.columns=columns
        self.build()
        self.saver = tf.train.Saver()

    def add_placeholders_op(self):
        state_size = Config.state_size
        self.s = tf.placeholder(tf.float32, (None, Config.rows,Config.columns,state_size))
        self.s_ = tf.placeholder(tf.float32, (None, Config.rows,Config.columns,state_size))
        self.a = tf.placeholder(tf.int32, (None))
        self.r = tf.placeholder(tf.float32, (None))


    def add_update_target_op(self, q_scope, target_q_scope):
        target_q_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        regular_q_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        update_target_op = [tf.assign(target_q_wts[i], regular_q_wts[i]) for i in range(len(target_q_wts))]
        self.update_target_op = tf.group(*update_target_op)

        ##############################################################
        ######################## END YOUR CODE #######################


    def deep_get_q_values_op(self, state, scope, reuse=False):
        num_actions = Config.num_actions
        out = state
        #print('state:',state)
        #print('num_actions',num_actions)
        with tf.variable_scope(scope, reuse=reuse) as _:
            #out = tf.contrib.layers.conv2d(out, num_outputs=32, kernel_size=[8,8], stride=4)
            #out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[4,4], stride=2)
            #out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[3,3], stride=1)

            #10x18
            out = tf.contrib.layers.conv2d(out, num_outputs=32, kernel_size=[3,3], stride=1)
            #8 x 16

            #
            out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[4,4], stride=2)
            #3 x 7
            out = tf.contrib.layers.conv2d(out, num_outputs=128, kernel_size=[3,3], stride=2)
            out = tf.contrib.layers.flatten(out)
            out = tf.contrib.layers.fully_connected(out, num_outputs=256)
            out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        ##############################################################
        ######################## END YOUR CODE #######################

        return out

    def add_loss_op(self, q, q_t):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = Config.num_actions
        upd = self.r + Config.gamma * tf.reduce_max(q_t, axis=1) - tf.reduce_sum(tf.one_hot(self.a, num_actions)*q, axis=1)
        self.loss = tf.reduce_mean(upd**2.)

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):

        optimizer = tf.train.AdamOptimizer(Config.lr)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        self.train_op = optimizer.minimize(self.loss, var_list=q_vars)

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

        self.q_t = self.deep_get_q_values_op(self.s_, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q,self.q_t)

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
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})
        return np.argmax(action_values, axis=1), action_values

    def get_epsilon(self, is_training):
        epsilon = 0.
        if is_training:
            epsilon = max(Config.epsilon_train_start + self.t*(Config.epsilon_train_end-Config.epsilon_train_start)/float(Config.epsilon_decay_steps), 
                Config.epsilon_train_end)
        else:
            epsilon = epsilon_soft
        return epsilon


    def get_action(self, states, is_training=True):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        best_actions = self.get_best_action(states)[0]
        random_actions = np.random.randint(0, Config.num_actions, len(states))
        probs = np.random.random(len(states))
        eps = self.get_epsilon(is_training)
        #print eps
        return np.where(probs < eps, random_actions, best_actions)

    def get_state(self, agent, obs):
        state = np.zeros((Config.rows,Config.columns,Config.state_size))
        team_players_start_index = 2
        for o in obs:
            #print(o)
            if o[0] == "loc":
                state[o[1][0]-1,o[1][1]-1,0]=1
            elif o[0] == "player":
                if agent.left_team == o[1][0] and agent.uni_number != o[1][1]:
                    state[o[1][2]-1,o[1][3]-1,1]=1
                elif agent.left_team != o[1][0]:
                    state[o[1][2]-1,o[1][3]-1,2]=1
            elif o[0] == "ball":
                state[o[1][0]-1,o[1][1]-1,3]=1
        return state


    def train(self):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """
        t = 0
        while True:
            state_prevs,action_prevs,reward_prevs,state_news = self.read_sars()
            
            time.sleep(2)
            if(len(state_prevs)>0):
                print('@_@ state_prevs',state_prevs.shape)
                print('@_@ state_news',state_news.shape)
                print('@_@ reward_prevs',reward_prevs.shape)
                print('@_@ action_prevs',action_prevs.shape)
                
                #print('@_@ state_prevs',state_prevs)
                #print('@_@ state_news',state_news)
                #print('@_@ reward_prevs',reward_prevs)
                #print('@_@ action_prevs',action_prevs)
                
                loss, _ = self.sess.run([self.loss, self.train_op],
                                        feed_dict={
                                            self.s: state_prevs,
                                            self.s_: state_news,
                                            self.r:  reward_prevs,
                                            self.a: action_prevs
                                        })

                print('Learner updated...')
                #Update model
                self.saver.save(self.sess, "tmp/model.ckpt")
                #os.rename('tmp/model.txt', 'tmp/model.ckpt')
                print('Model saved...')
                
            
                
            if self.t % Config.target_update_freq == 0:
                self.update_target_params()
                #print('final score:',self.episode_team_goals,self.episode_opp_goals)
                self.episode_opp_goals=0
                self.episode_team_goals=0
            self.t += 1

     
    def reward(self, state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new):
        preAmIBallOwner = np.sum(state_prev[:, :, 0]*state_prev[:, :, 3]) != 0
        preAreWeBallOwner = preAmIBallOwner or np.sum(state_prev[:, :, 1]*state_prev[:, :, 3]) != 0
        curAmIBallOwner = np.sum(state_new[:, :, 0]*state_new[:, :, 3]) != 0
        curAreWeBallOwner = curAmIBallOwner or np.sum(state_new[:, :, 1]*state_new[:, :, 3]) != 0
        isCollaborative = Config.collaborative_rewards

        #if True:
        #    return (self.episode_opp_goals+1-self.episode_team_goals)*Config.RewardLooseMatch/(1+Config.episode_len -
        #                                                                                       (self.t%Config.episode_len))
           
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

        if action_prev != 0 and action_prev != 9 and np.sum(state_prev[:,:,0]*state_new[:,:,0]) != 0:
            return Config.RewardIllegalMovment

        return Config.RewardEveryMovment

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


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
        
    def read_sars(self):
        s = []
        a = []
        r =[]
        s_ = []
        onlyfiles = [f for f in os.listdir(Config.sars_path) if os.path.isfile(os.path.join(Config.sars_path, f))]
        for file_path in onlyfiles:
            param_file = os.path.join(Config.sars_path,file_path)
            if param_file.endswith('txt'):
                with open(param_file, 'r') as f:
                    sars_dic = pickle.load(f)
                    if('s' in sars_dic):
                        s.append(sars_dic['s'])
                        a.append(sars_dic['a'])
                        r.append(sars_dic['r'])
                        s_.append(sars_dic['s_'])
                        
                #print('param_file::',param_file)
                os.remove(os.path.abspath(param_file))
        if(len(s)>0):
            return (np.array(s),np.array(a),np.array(r),np.array(s_))
        else:
            return ([],[],[],[])

