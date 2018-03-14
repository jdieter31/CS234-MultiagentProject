# coding: utf-8
'''
IMPORTANT: PLAYERS MUST BE NUMBERED 0, 1, 2, ...

'''
# In[1]:

import socket
import re
import select
import sys
import numpy as np
from AgentInterface import *
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers
from multiprocessing.dummy import Pool as ThreadPool


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# In[2]:                                                                

class Config():

	load_model = False 
	save_model = True


	model_name = 'Counterfactual'
	model_dir = 'models/' + model_name + '/'
	rows = 10
	columns = 18
	epsilon_train_start = 0.5
	epsilon_train_end = 0.1
	epsilon_decay_steps = 100000
	epsilon_soft = 0.05
	gamma = 0.9
	lr = 0.001
	num_players_per_team = 3
	num_actions = 9 + num_players_per_team - 1
	state_size = num_players_per_team + 2
	num_train_episodes = 200
	num_eval_episodes = 0
	max_episode_length = 300

	episode_len = 50
	# target_update_freq = episode_len

	# RewardLooseMatch = -10.
	# RewardWinMatch = 10.
	target_update_freq = 300

	target_update_freq = 50
	
	RewardEveryMovment = -2.
	RewardSuccessfulPass = -2.
	RewardHold = -2.
	RewardIllegalMovment = -3.
	RewardTeamCatchBall = 10.
	RewardTeamLooseBall = -10.
	RewardSelfCatchBall = 10.
	RewardSelfLooseBall = -10.
	RewardTeamScoreGoal = 50.
	RewardSelfScoreGoal = 50.
	RewardTeamRecvGoal = -50.
	RewardTeamOwnGoal = -75.
	RewardSelfOwnGoal = -100.
	RewardOpponentOwnGoal = 10.
	collaborative_rewards = True

	batch_size = 5

	lambd = 0.8


# In[3]:


class QN:
	q_t = None
	t = 0
	# episode_team_goals=0
	# episode_opp_goals=0
	"""
	Abstract Class for implementing a Q Network
	"""

	def __init__(self, rows=10, columns=18):
		Config.rows=rows
		Config.columns=columns
		self.build()
		self.save_dir = Config.model_dir
		self.saver = tf.train.Saver()

	def add_placeholders_op(self):
		state_size = Config.state_size
		self.s = tf.placeholder(tf.float32, (None, Config.rows, Config.columns, state_size))
		self.s_ = tf.placeholder(tf.float32, (None, Config.rows, Config.columns, state_size))
		self.a_minus = tf.placeholder(tf.int32, (None, Config.num_players_per_team - 1))
		self.TD_targets = tf.placeholder(tf.float32, (None))
		self.agent = tf.placeholder(tf.int32, (None))
		self.a = tf.placeholder(tf.int32, (None))
		self.r = tf.placeholder(tf.float32, (None))
		self.advantages_in = tf.placeholder(tf.float32, (None))

	def add_update_target_op(self, q_scope, target_q_scope):
		target_q_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
		regular_q_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
		update_target_op = [tf.assign(target_q_wts[i], regular_q_wts[i]) for i in range(len(target_q_wts))]
		self.update_target_op = tf.group(*update_target_op)

	##############################################################
	######################## END YOUR CODE #######################

	def deep_get_q_values_op(self, state, scope, a_minus, agent, reuse=False):
		num_actions = Config.num_actions
		out = state
		# print('state:',state)
		# print('num_actions',num_actions)
		with tf.variable_scope(scope, reuse=reuse) as _:
			# out = tf.contrib.layers.conv2d(out, num_outputs=32, kernel_size=[8,8], stride=4)
			# out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[4,4], stride=2)
			# out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[3,3], stride=1)

			# 10x18
			out = tf.contrib.layers.conv2d(out, num_outputs=32, kernel_size=[3, 3], stride=1)
			# 8 x 16

			#
			out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[4, 4], stride=2)
			# 3 x 7
			out = tf.contrib.layers.conv2d(out, num_outputs=128, kernel_size=[3, 3], stride=2)
			out = tf.contrib.layers.flatten(out)
			a_minus_one_hot = tf.reshape(tf.one_hot(a_minus, num_actions), [-1, num_actions*(Config.num_players_per_team-1)])
			agent_one_hot = tf.reshape(tf.one_hot(agent, Config.num_players_per_team), [-1, Config.num_players_per_team])
			out = tf.concat([out, a_minus_one_hot, agent_one_hot], axis=1)
			out = tf.contrib.layers.fully_connected(out, num_outputs=256)
			out = tf.contrib.layers.fully_connected(out, num_outputs=128)
			out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
		##############################################################
		######################## END YOUR CODE #######################

		return out

	def deep_get_policy_op(self, state, agent, scope, reuse=False):
		num_actions = Config.num_actions
		out = state
		# print('state:',state)
		# print('num_actions',num_actions)
		with tf.variable_scope(scope, reuse=reuse) as _:
			# out = tf.contrib.layers.conv2d(out, num_outputs=32, kernel_size=[8,8], stride=4)
			# out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[4,4], stride=2)
			# out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[3,3], stride=1)

			# 10x18
			out = tf.contrib.layers.conv2d(out, num_outputs=32, kernel_size=[3, 3], stride=1)
			# 8 x 16

			#
			out = tf.contrib.layers.conv2d(out, num_outputs=64, kernel_size=[4, 4], stride=2)
			# 3 x 7
			out = tf.contrib.layers.conv2d(out, num_outputs=128, kernel_size=[3, 3], stride=2)
			out = tf.contrib.layers.flatten(out)
			agent_one_hot = tf.reshape(tf.one_hot(agent, Config.num_players_per_team), [-1, Config.num_players_per_team])
			out = tf.concat([out, agent_one_hot], axis=1)
			out = tf.contrib.layers.fully_connected(out, num_outputs=256)
			out = tf.contrib.layers.fully_connected(out, num_outputs=128)
			out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
		##############################################################
		######################## END YOUR CODE #######################

		return out

	def add_loss_op(self, q, TD_Targets):
		"""
		Sets the loss of a batch, self.loss is a scalar

		"""
		upd = tf.reduce_sum(TD_Targets - tf.reduce_sum(tf.one_hot(self.a, Config.num_actions) * q, axis=1))
		self.loss = tf.reduce_mean(upd ** 2.)

	##############################################################
	######################## END YOUR CODE #######################

	def get_optimizer_op(self, scope):

		optimizer = tf.train.AdamOptimizer(Config.lr)
		q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		return optimizer.minimize(self.loss, var_list=q_vars)

	##############################################################
	######################## END YOUR CODE #######################

	def add_policy_op(self, scope):
		optimizer = tf.train.AdamOptimizer(Config.lr)
		policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		loss = -tf.reduce_sum(self.logprob*self.advantages_in)
		return optimizer.minimize(loss, var_list=policy_vars)


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
		self.q = self.deep_get_q_values_op(self.s, scope="q", a_minus=self.a_minus, agent=self.agent, reuse=False)

		self.q_t = self.deep_get_q_values_op(self.s, scope="target_q", a_minus=self.a_minus, agent=self.agent,
		                                     reuse=False)

		self.policy = self.deep_get_policy_op(self.s, agent=self.agent, scope="pi", reuse=False)
		self.sampled_action = tf.squeeze(tf.multinomial(self.policy, 1), axis=1)
		self.logprob = tf.reshape(-tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy, labels=self.a),
		                          shape=[-1, 1])

		self.advantage = tf.reduce_sum(tf.one_hot(self.a, Config.num_actions) * self.q, axis=1) - tf.reduce_sum(tf.exp(self.logprob) * self.q, axis=1)


		# add update operator for target network
		self.add_update_target_op("q", "target_q")

		# add square loss
		self.add_loss_op(self.q, self.TD_targets)

		# add optmizer for the main networks
		self.train_op_global = self.get_optimizer_op("q")

		self.policy_op = self.add_policy_op("pi")

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

	def get_epsilon(self, is_training):
		epsilon = 0.
		if is_training:
			epsilon = max(
				Config.epsilon_train_start + self.t * (Config.epsilon_train_end - Config.epsilon_train_start) / float(
					Config.epsilon_decay_steps),
				Config.epsilon_train_end)
		else:
			epsilon = Config.epsilon_soft
		return epsilon

	def sample_action(self, state, a):
		return self.sess.run([self.sampled_action], feed_dict={
			self.s: [state],
			self.agent: [a]
		})[0]

	def get_state(self, agent, obs):
		state = np.zeros((Config.rows, Config.columns, Config.state_size))
		for o in obs:
			# print(o)
			if o[0] == "loc":
				a = agent.uni_number + 1
				state[o[1][0] - 1, o[1][1] - 1, a] = 1
				print(o)
			elif o[0] == "player":
				if agent.left_team == o[1][0] and agent.uni_number != o[1][1]:
					a = o[1][1] + 1
					state[o[1][2] - 1, o[1][3] - 1, a] = 1
				elif agent.left_team != o[1][0]:
					state[o[1][2] - 1, o[1][3] - 1, 1] = 1
				print(o)
			elif o[0] == "ball":
				state[o[1][0] - 1, o[1][1] - 1, 0] = 1
				print(o)
		print(state)
		return state

	def sample_episode(self, initial_state):
		i = 0
		completed = False
		won = False
		score_team_prev = [0 for a in range(Config.num_players_per_team)]
		score_opp_prev = [0 for a in range(Config.num_players_per_team)]
		actions = []
		states = [initial_state]
		rewards = []
		actions_chosen = 0
		'''
		# Send initial action
		for a in range(Config.num_players_per_team):
			agent = self.agents[a]
			actions[0][a] = self.sample_action(states[0], a)
			if actions[0][a] <= 8:
				agent.send_action("move", actions[0][a])
			else:
				if actions[0][a] - 9 < a:
					agent.send_action("pass", actions[0][a] - 8)
				else:
					agent.send_action("pass", actions[0][a] - 7)
		'''

		while not completed:
			for a in range(Config.num_players_per_team):
				new_cycle = False
				agent = self.agents[a]
				obs = []
				if i != 0:
					self.agent_obs[a] = agent.observe_from_server()
					obs = self.agent_obs[a]
				for o in obs:
					if o[0] == "cycle":
						new_cycle = True
						break
				if new_cycle or i == 0:
					# for a in range(Config.num_players_per_team):
					if len(states) < i + 1:
						states.append(self.get_state(agent, obs))
					if i != 0:
						score = None
						for o in obs:
							if o[0] == "score":
								if agent.left_team:
									score = [o[1][0], o[1][1]]
								else:
									score = [o[1][1], o[1][0]]
						score_team_new, score_opp_new = score[0], score[1]
					if i > 1:
						if (score_team_new, score_opp_new) != (score_team_prev[a], score_opp_prev[a]):
							completed = True
						if a == score_team_new > score_team_prev[a]:
							won = True
						if a == 0:
							rewards.append(self.reward(states[i - 1], states[i], actions[i - 1][a], score_team_prev[a],
							                           score_opp_prev[a], score_team_new, score_opp_new))
					# --print(reward_prev)
					# self.episode_opp_goals+=score_opp_new-score_opp_prev
					# self.episode_team_goals+=score_team_new-score_team_prev

					if completed:
						break

					if len(actions) < i + 1:
						actions.append(np.zeros((Config.num_players_per_team), dtype=np.int32))
					actions[i][a] = self.sample_action(states[i], a)
					actions_chosen += 1
					# if self.i > Config.max_episode_length and agent.uni_number == 0:
					#     agent.send_action("restart", False)
					#     self.i = 0
					#     num_episodes += 1
					# else:
					if actions[i][a] <= 8:
						agent.send_action("move", actions[i][a])
					else:
						if actions[i][a] - 9 < a:
							agent.send_action("pass", actions[i][a] - 8)
						else:
							agent.send_action("pass", actions[i][a] - 7)

					if i != 0:
						score_team_prev[a] = score_team_new
						score_opp_prev[a] = score_opp_new
					if actions_chosen == Config.num_players_per_team:
						i += 1
						actions_chosen = 0
		return np.array(states[:-1]), np.array(actions), np.array(rewards), states[-1], won

	def train(self):
		"""
		Performs training of Q

		Args:
			exp_schedule: Exploration instance s.t.
				exp_schedule.get_action(best_action) returns an action
			lr_schedule: Schedule for learning rate
		"""

		# initialize replay buffer and variables
		num_episodes = 0
		self.agents = []
		self.agent_obs = []
		# agents = []
		# agent_obs = []
		for a in range(Config.num_players_per_team):
			agent = AgentInterface('COMA', a + 1)
			agent.set_home(int(Config.rows / 2) - int(Config.num_players_per_team / 2) + 1 + a, 2)
			self.agents.append(agent)
			obs = agent.observe_from_server()
			self.agent_obs.append(obs)

		initial_state = None
		for a in range(Config.num_players_per_team):
			while ("start", 0) not in self.agent_obs[a]:
				self.agent_obs[a] = self.agents[a].observe_from_server()

		inital_obs = 0
		while inital_obs < Config.num_players_per_team:
			for a in range(Config.num_players_per_team):
				new_cycle = False
				agent = self.agents[a]
				self.agent_obs[a] = agent.observe_from_server()
				obs = self.agent_obs[a]
				for o in obs:
					if o[0] == "cycle":
						new_cycle = True
						break
				if new_cycle:
					initial_state = self.get_state(agent, obs)
					inital_obs += 1
		i = 0
		episodes = 0
		goals_scored = 0
		while True:
			states = None
			actions = None
			TDTargets = None
			for i in range(Config.batch_size):
				estates, eactions, erewards, next_state, won = self.sample_episode(initial_state)
				episodes += 1
				if won:
					goals_scored += 1
				if episodes % Config.num_train_episodes == 0:
					print "TRAIN PROPORTION OF EPISODES WON %s" % (float(goals_scored) / Config.num_train_episodes)

					if Config.save_model:
						self.save()
					episodes = 0
					goals_scored = 0

				initial_state = next_state
				batch_targets = self.getTDTargets(estates, eactions, erewards)
				if TDTargets is None:
					TDTargets = batch_targets
				else:
					TDTargets = np.append(TDTargets, batch_targets, axis=0)
				if states is None:
					states = estates
				else:
					states = np.append(states, estates, axis=0)
				if actions is None:
					actions = eactions
				else:
					actions = np.append(actions, eactions, axis=0)

			i += len(states)

			for a in range(Config.num_players_per_team):
				a_minus = np.delete(actions, a, 1)
				agent_actions = actions[:, a]
				loss, _ = self.sess.run([self.loss, self.train_op_global], feed_dict={
					self.s: states,
					self.a: agent_actions,
					self.a_minus: a_minus,
					self.TD_targets: TDTargets[:,a],
					self.agent: [a for _ in range(states.shape[0])]})

			if i > Config.target_update_freq:
				self.update_target_params()
				i = 0

			for a in range(Config.num_players_per_team):
				a_minus = np.delete(actions, a, 1)
				agent_actions = actions[:, a]
				advantages = self.sess.run([self.advantage], feed_dict={
					self.s: states,
					self.a: agent_actions,
					self.a_minus: a_minus,
					self.agent: [a for _ in range(states.shape[0])]
				})[0]
				self.sess.run([self.policy_op], feed_dict={
					self.s: states,
					self.a: agent_actions,
					self.agent: [a for _ in range(states.shape[0])],
					self.advantages_in: advantages
				})


	def getTDTargets(self, states, actions, rewards):
		TDTargets = np.zeros((states.shape[0], Config.num_players_per_team))
		for a in range(Config.num_players_per_team):
			a_minus = np.delete(actions, a, 1)
			aTargetQs = self.sess.run([self.q_t], feed_dict={
				self.s: states,
				self.a_minus: a_minus,
				self.agent: [a for _ in range(len(states))]
			})[0]
			one_hot = np.eye(Config.num_actions)[actions[:, a]]
			aTargetQ = np.sum(aTargetQs*one_hot, axis=1)
			extendedATargetQ = np.append(aTargetQ, np.zeros(aTargetQ.shape), axis=0)
			extendedRewards = np.append(rewards, np.zeros(rewards.shape), axis=0)
			nReturns = np.zeros((states.shape[0], states.shape[0]))
			for n in range(1, states.shape[0]):
				nReturn = np.array(
					[np.dot(extendedRewards[i:i + n], np.geomspace(1, Config.gamma ** (n - 1), num=n)) for i in
					 range(states.shape[0])])
				nReturn = nReturn + (Config.gamma ** n) * extendedATargetQ[n:n + states.shape[0]]
				nReturns[:, n - 1] = nReturn
			TDTargets[:, a] = (1 - Config.lambd) * np.dot(nReturns, np.geomspace(1, Config.lambd ** (states.shape[0] - 1), num=states.shape[0]))
		return TDTargets

	def reward(self, state_prev, state_new, action_prev, score_team_prev, score_opp_prev, score_team_new, score_opp_new):
		preAreWeBallOwner = np.sum(state_prev[:, :, 1] * state_prev[:, :, 1]) == 0
		curAreWeBallOwner = np.sum(state_new[:, :, 1] * state_new[:, :, 1]) == 0

		# if True:
		#    return (self.episode_opp_goals+1-self.episode_team_goals)*Config.RewardLooseMatch/(1+Config.episode_len -
		#                                                                                       (self.t%Config.episode_len))

		# if we scored a goal
		if score_team_new > score_team_prev:
			return Config.RewardTeamScoreGoal

		# if we received a goal
		if score_opp_new > score_opp_prev:
			return Config.RewardTeamRecvGoal

		if curAreWeBallOwner and (not preAreWeBallOwner):
			return Config.RewardTeamCatchBall
		if (not curAreWeBallOwner) and preAreWeBallOwner:
			return Config.RewardTeamLooseBall

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


if __name__ == '__main__':
	model = QN()
	model.run()
