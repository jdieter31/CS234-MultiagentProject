
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
import random
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
    epsilon_train_end = 0.1
    epsilon_decay_steps = 100000
    epsilon_soft = 0.05
    gamma = 0.9
    lr = 0.001
    num_group_actions = 3
    num_players_per_team = 3
    num_actions = 9 + num_players_per_team - 1
    state_size = 4 + num_group_actions
    
    replay_buffer_size = 50000
    mini_batch_size = 1000
    
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



class QNParamSharingComLeaner():
    player = 0
    q_t=None
    t = 0
    episode_team_goals=0
    episode_opp_goals=0
    replay_buffer_s =[]
    replay_buffer_a =[]
    replay_buffer_r =[]
    replay_buffer_s_ =[]
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
            out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions*Config.num_group_actions, activation_fn=None)
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
        upd = self.r + Config.gamma * tf.reduce_max(q_t, axis=1) - tf.reduce_sum(tf.one_hot(self.a, num_actions * Config.num_group_actions)*q, axis=1)
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
                
                self.replay_buffer_s.extend(state_prevs.tolist())
                self.replay_buffer_a.extend(action_prevs.tolist())
                self.replay_buffer_r.extend(reward_prevs.tolist())
                self.replay_buffer_s_.extend(state_news.tolist())
                
                c = list(zip(self.replay_buffer_s,self.replay_buffer_a,self.replay_buffer_r,self.replay_buffer_s_))
                random.shuffle(c)
                self.replay_buffer_s,self.replay_buffer_a,self.replay_buffer_r,self.replay_buffer_s_ = zip(*c)
                
                
                self.replay_buffer_s=list(self.replay_buffer_s[0:Config.replay_buffer_size])
                self.replay_buffer_a=list(self.replay_buffer_a[0:Config.replay_buffer_size])
                self.replay_buffer_r=list(self.replay_buffer_r[0:Config.replay_buffer_size])
                self.replay_buffer_s_=list(self.replay_buffer_s_[0:Config.replay_buffer_size])
                
                
                state_prevs=self.replay_buffer_s[0:Config.mini_batch_size]
                action_prevs=self.replay_buffer_a[0:Config.mini_batch_size]
                reward_prevs=self.replay_buffer_r[0:Config.mini_batch_size]
                state_news=self.replay_buffer_s_[0:Config.mini_batch_size]
                
                #print('@_@ state_prevs',state_prevs)
                #print('@_@ state_news',state_news)
                #print('@_@ reward_prevs',reward_prevs)
                #print('@_@ action_prevs',action_prevs)
                
                try:
                    loss, _ = self.sess.run([self.loss, self.train_op],
                                            feed_dict={
                                                self.s: state_prevs,
                                                self.s_: state_news,
                                                self.r:  reward_prevs,
                                                self.a: action_prevs
                                            })
                except Exception:
                    print('loss computation failed...')

                print('Learner updated...')
                #Update model
                self.saver.save(self.sess, "tmp/model.ckpt")
                #os.rename('tmp/model.txt', 'tmp/model.ckpt')
                print('Model saved...')
                
            
                
            if self.t % Config.target_update_freq == 0:
                self.update_target_params()
                
            self.t += 1

     
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
        #try:
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
            s=np.array(s)
            a=np.array(a)
            r=np.array(r)
            s_=np.array(s_)
            return (s,a,r,s_)
        else:
            return ([],[],[],[])

