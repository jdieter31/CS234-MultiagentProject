# coding: utf-8
'''
IMPORTANT: PLAYERS MUST BE NUMBERED 0, 1, 2, ...

'''
# In[1]:

import socket
import re
import select
import sys
import os
import time
import pickle
import numpy as np
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers
from multiprocessing.dummy import Pool as ThreadPool 
import os


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

# In[2]:

class Config():
    load_model = False
    save_model = True

    model_name = 'DQN_PS_Com_min_batch_diff_learn'
    model_dir = 'models/' + model_name + '/' 
        
    num_group_actions = 3
    group_strategy_dir = './strategy/'
    if not os.path.exists(group_strategy_dir):
        os.makedirs(group_strategy_dir)
    
    sars_path = './param_sharing'
    if not os.path.exists(sars_path):
        os.makedirs(sars_path)
    
    rows = 10
    columns = 18
    epsilon_train_start = 0.5
    epsilon_train_end = 0.05
    epsilon_decay_steps = 300000
    epsilon_soft = 0.05
    gamma = 0.9
    lr = 0.001
    num_players_per_team = 3
    num_actions = 9 + num_players_per_team - 1
    state_size = 4 + num_group_actions
    num_train_episodes = 200
    num_eval_episodes = 0
    max_episode_length = 300

    episode_len = 50
    # target_update_freq = episode_len

    # RewardLooseMatch = -10.
    # RewardWinMatch = 10.
    target_update_freq = 50
    model_update_freq = 50

    RewardEveryMovment = 0.#-2.
    RewardSuccessfulPass = 0.#-2.
    RewardHold = 0.#-2.
    RewardIllegalMovment = 0.#v-2.
    RewardTeamCatchBall = 0.#v10.
    RewardTeamLooseBall = 0.#-10.
    RewardSelfCatchBall = 0.#10.
    RewardSelfLooseBall = 0.#-10.
    RewardTeamScoreGoal = 1.#50.
    RewardSelfScoreGoal = 1.#50.
    RewardTeamRecvGoal = -1.#-50.
    RewardTeamOwnGoal = -1.#-75.
    RewardSelfOwnGoal = -1.#-100.
    RewardOpponentOwnGoal = 1.#10.
    collaborative_rewards = True


    
class QNParamSharingComAgent:
    player = 0
    q_t=None
    t = 0
    # episode_team_goals=0
    # episode_opp_goals=0
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self,player=0,rows=10,columns=18):
        self.player = player
        Config.rows=rows
        Config.columns=columns
        self.build()
        self.save_dir = Config.model_dir + str(self.player) + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.saver = tf.train.Saver()
        self.last_model_update = None
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
        save_path = self.save_dir + 'model.weights'
        print 'SAVING TO %s' % save_path
        self.saver.save(self.sess, save_path)


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
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if Config.load_model and ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            print "Reading model parameters from %s" % ckpt.model_checkpoint_path
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print "Created model with fresh parameters."
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        # # tensorboard stuff
        # self.add_summary()

        # initiliaze all variables
        


    def get_best_action(self, state):
        """
        Returns best action according to the network

        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        action_values_joint = np.reshape(action_values,(Config.num_actions,Config.num_group_actions))
        #print('action_values',action_values.shape)
        best_joint_action = np.argmax(action_values)
        #print('best_joint_action',best_joint_action.shape)
        return best_joint_action, action_values
            
            
    def get_epsilon(self, is_training):
        epsilon = 0.
        if is_training:
            epsilon = max(Config.epsilon_train_start + self.t*(Config.epsilon_train_end-Config.epsilon_train_start)/float(Config.epsilon_decay_steps), 
                Config.epsilon_train_end)
        else:
            epsilon = Config.epsilon_soft
        return epsilon


    def get_action(self, states, is_training=True):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        best_joint_action,action_values = self.get_best_action(states)
        random_actions = np.random.randint(0, Config.num_actions * Config.num_group_actions)
        probs = np.random.random()
        eps = self.get_epsilon(is_training)
        
        #print('best_joint_action',best_joint_action)
        #print('random_actions',random_actions)
        
        if(probs < eps):
            joint_action = random_actions
        else:
            joint_action = best_joint_action
        #joint_action = np.where(probs < eps, random_actions, best_joint_action)
        
        group_action = joint_action // Config.num_actions
        individual_action = joint_action % Config.num_actions
        
        self.save_group_action(group_action)
        # print eps
        return individual_action,action_values[joint_action]

    def save_group_action(self,best_group_action):
        with open(Config.group_strategy_dir+str(self.player), 'w') as f:
            f.write(str(best_group_action))
            
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
                    state[o[1][2]-1,o[1][3]-1,2 + self.get_group_strategy(o[1][1])]=1
                    #state[o[1][2]-1,o[1][3]-1,1]=1
                elif agent.left_team != o[1][0]:
                    state[o[1][2]-1,o[1][3]-1,-2]=1
            elif o[0] == "ball":
                state[o[1][0]-1,o[1][1]-1,-1]=1
        return state

    def get_group_strategy(self,player):
        if os.path.exists(Config.group_strategy_dir+str(player)):
            with open(Config.group_strategy_dir+str(player), 'r') as f:
                strategy = f.read()
            if (len(strategy)!=0):
                try:
                    return int(strategy) 
                except Exception:
                    return 0
            else:
                return 0 # default strategy
        else:
            return 0
        
    def train(self):
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
        num_episodes = 0
        # agents = []
        # agent_obs = []
        # for a in range(Config.num_players_per_team):
        agent = AgentInterface(Config.model_name, self.player)
        agent.set_home(int(Config.rows/2) - int(Config.num_players_per_team/2) + self.player, 2)
        # agents.append(agent)
        obs = agent.observe_from_server()
        # agent_obs.append(obs)
        self.i = 0
        state_prev = None
        action_prev = None
        train_episodes_won = 0
        # eval_episodes_won = 0
        last_print_episode = 0
        last_save_episode = 0
        # for a in range(Config.num_players_per_team):
        while ("start", 0) not in obs:
            obs = agent.observe_from_server()
        while True:
            obs = agent.observe_from_server()
            new_cycle = False
            for o in obs:
                if o[0] == "cycle":
                    new_cycle = True
                    break
            if new_cycle:
                if num_episodes % (Config.num_train_episodes + Config.num_eval_episodes) == 0 and num_episodes > last_print_episode:
                    if self.player == 1:
                        print "NUMBER OF TRAINING ITERATIONS: %d" % (self.t)
                        last_print_episode = num_episodes
                        print "TRAIN PROPORTION OF EPISODES WON %s" % (float(train_episodes_won) / Config.num_train_episodes)
                        # print "EVAL. PROPORTION OF EPISODES WON %s" % (float(eval_episodes_won) / Config.num_eval_episodes)
                        with open(Config.model_name + '.out', 'a') as f:
                            f.write('%s,%s\n' % (self.t, float(train_episodes_won) / Config.num_train_episodes))
                        train_episodes_won = 0
                    if Config.save_model and num_episodes > last_save_episode:
                        self.save()
                        last_save_episode = num_episodes
                    # eval_episodes_won = 0



                if num_episodes % (Config.num_train_episodes + Config.num_eval_episodes) < Config.num_train_episodes: #TRAINING
                    # for a in range(Config.num_players_per_team):
                    state_new = self.get_state(agent, obs)
                    if action_prev is not None:
                        score = None
                        for o in obs:
                            if o[0] == "score":
                                if agent.left_team:
                                    score = [o[1][0], o[1][1]]
                                else:
                                    score = [o[1][1], o[1][0]]
                        score_team_new, score_opp_new = score[0], score[1]
                        if (score_team_new, score_opp_new) != (score_team_prev, score_opp_prev):
                            self.i = 0
                            num_episodes += 1
                        if score_team_new > score_team_prev:
                            train_episodes_won += 1

                        reward_prev = self.reward(state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new)
                        #--print(reward_prev)
                        # self.episode_opp_goals+=score_opp_new-score_opp_prev
                        # self.episode_team_goals+=score_team_new-score_team_prev
                        
                        self.write_sars(state_prev,state_new,reward_prev,action_prev,self.t,action_prev_action_value,self.player)
                        
                        #Update model
                        if self.t % Config.model_update_freq == 0 and os.path.exists('tmp/model.ckpt.data-00000-of-00001') and self.last_model_update!=os.path.getmtime('tmp/model.ckpt.data-00000-of-00001') :
                            self.last_model_update = os.path.getmtime('tmp/model.ckpt.data-00000-of-00001')
                            try:
                                self.saver.restore(self.sess, "tmp/model.ckpt")
                                print('load model successful')
                            except Exception:
                                print('load model failed')
                        
                        self.t += 1
                        score_team_prev = score_team_new
                        score_opp_prev = score_opp_new
                    action_new,action_new_action_value = self.get_action(state_new, True)
                    # if self.i > Config.max_episode_length and agent.uni_number == 0:
                    #     agent.send_action("restart", False)
                    #     self.i = 0                        
                    #     num_episodes += 1
                    # else:
                    if action_new <= 8:
                        agent.send_action("move", action_new)
                    else:
                        if action_new - 8 < self.player:
                            agent.send_action("pass", action_new-8)
                        else:
                            agent.send_action("pass", action_new-7)
                    state_prev = state_new.copy()
                    action_prev = action_new
                    action_prev_action_value = action_new_action_value
                # else: # EVALUATION
                #     # for a in range(Config.num_players_per_team):
                #     state_new = self.get_state(agent, obs)
                #     if action_prev is not None:
                #         score = None
                #         for o in obs:
                #             if o[0] == "score":
                #                 if agent.left_team:
                #                     score = [o[1][0], o[1][1]]
                #                 else:
                #                     score = [o[1][1], o[1][0]]
                #         score_team_new, score_opp_new = score[0], score[1]
                #         if (score_team_new, score_opp_new) != (score_team_prev, score_opp_prev):
                #             self.i = 0
                #             num_episodes += 1
                #         if score_team_new > score_team_prev:
                #             eval_episodes_won += 1
                #         reward_prev = self.reward(state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new)

                #         #--print(reward_prev)
                #         # self.episode_opp_goals+=score_opp_new-score_opp_prev
                #         # self.episode_team_goals+=score_team_new-score_team_prev
                        
                #         score_team_prev = score_team_new
                #         score_opp_prev = score_opp_new
                #     action_new = self.get_action(state_new, False)[0]
                #     # if self.i > Config.max_episode_length and agent.uni_number == 0:
                #     #     agent.send_action("restart", False)
                #     #     self.i = 0
                #     #     num_episodes += 1
                #     # else:
                #     if action_new <= 8:
                #         agent.send_action("move", action_new)
                #     else:
                #         if action_new - 8 < self.player:
                #             agent.send_action("pass", action_new-8)
                #         else:
                #             agent.send_action("pass", action_new-7)
                #     state_prev = state_new.copy()
                #     action_prev = action_new
                self.i += 1

     
    def reward(self, state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new):
        preAmIBallOwner = np.sum(state_prev[:, :, 0]*state_prev[:, :, -1]) != 0
        preAreWeBallOwner = preAmIBallOwner or np.sum(state_prev[:, :, -2]*state_prev[:, :, -1]) == 0
        curAmIBallOwner = np.sum(state_new[:, :, 0]*state_new[:, :, -1]) != 0
        curAreWeBallOwner = curAmIBallOwner or np.sum(state_new[:, :, -2]*state_new[:, :, -1]) == 0
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
        
    def write_sars(self,s,s_,r,a,t,Q,i):
        param_file = os.path.join(Config.sars_path,str(time.time())+'_'+str(i))
        with open(param_file+'.tmp','w') as f:
            pickle.dump({
                's':s,
                'a':a,
                'r':r,
                's_':s_,
                't':t,
                'Q':Q
            }, f, pickle.HIGHEST_PROTOCOL)
        #print('param_file',param_file)
        os.rename(param_file+'.tmp', param_file+'.txt')
        


if __name__ == '__main__':
    model = QN(player=int(sys.argv[1]))
    model.run()
