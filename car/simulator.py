import numpy as np
from collections import defaultdict
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Config():
	epsilon_train = 0.05
	gamma = 1.0
	lr = 0.01
	max_timesteps = 300


class Car:
	# start_pos: 'N', 'S', 'E', 'W'
	# end_pos: 'N', 'S', 'E', 'W'
	def __init__(self, start_pos, end_pos, start_speed, lane_width, road_length):
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.lane_width = lane_width
		self.road_length = road_length
		self.dist = 0.
		self.speed = start_speed
		self.acceleration = 0.
		self.angle = None
		self.pos_x = None
		self.pos_y = None
		self.vel_x = None
		self.vel_y = None
		self.acc_x = 0.
		self.acc_y = 0.
		if start_pos == 'N':
			self.pos_x = -0.5*lane_width
			self.pos_y = lane_width + road_length
			self.vel_x = 0.
			self.vel_y = -start_speed
			self.angle = 3.*np.pi/2.		
		elif start_pos == 'S':
			self.pos_x = 0.5*lane_width
			self.pos_y = -lane_width - road_length
			self.vel_x = 0.
			self.vel_y = start_speed	
			self.angle = np.pi/2.
		elif start_pos == 'E':
			self.pos_x = lane_width + road_length
			self.pos_y = 0.5*lane_width
			self.vel_x = -start_speed
			self.vel_y = 0.
			self.angle = np.pi
		else:
			self.pos_x = -lane_width - road_length
			self.pos_y = -0.5*lane_width
			self.vel_x = start_speed
			self.vel_y = 0.
			self.angle = 0.
		self.completed_task = False

	# dist along path -> pos_x, pos_y, angle
	def update_pos_angle(self):
		if self.start_pos == 'N' and self.end_pos == 'N':
			if self.dist <= self.road_length:
				self.pos_x = -0.5*self.lane_width
				self.pos_y = self.lane_width + self.road_length - self.dist
				self.angle = 3.*np.pi/2.
			elif self.road_length < self.dist < self.road_length + 0.5*np.pi*self.lane_width:
				self.angle = 3.*np.pi/2. + (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = 0.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = self.lane_width + 0.5*self.lane_width*np.sin(self.angle - np.pi/2.)
			elif self.road_length + 0.5*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.5*np.pi*self.lane_width:
				self.pos_x = 0.5*self.lane_width
				self.pos_y = self.lane_width + self.dist - (self.road_length + 0.5*np.pi*self.lane_width)
				self.angle = np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'N' and self.end_pos == 'S':
			if self.dist < 2.*self.road_length + 2.*self.lane_width:				
				self.pos_x = -0.5*self.lane_width
				self.pos_y = self.lane_width + self.road_length - self.dist
				self.angle = 3.*np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'N' and self.end_pos == 'E':			
			if self.dist <= self.road_length:
				self.pos_x = -0.5*self.lane_width
				self.pos_y = self.lane_width + self.road_length - self.dist
				self.angle = 3.*np.pi/2.
			elif self.road_length < self.dist < self.road_length + 0.75*np.pi*self.lane_width:
				self.angle = 3.*np.pi/2. + (self.dist - self.road_length) / (1.5*self.lane_width)
				self.pos_x = self.lane_width + 1.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = self.lane_width + 1.5*self.lane_width*np.sin(self.angle - np.pi/2.)				
			elif self.road_length + 0.75*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.75*np.pi*self.lane_width:
				self.pos_x = self.lane_width + self.dist - (self.road_length + 0.75*np.pi*self.lane_width)
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			else:
				self.completed_task = True
		elif self.start_pos == 'N' and self.end_pos == 'W':			
			if self.dist <= self.road_length:
				self.pos_x = -0.5*self.lane_width
				self.pos_y = self.lane_width + self.road_length - self.dist
				self.angle = 3.*np.pi/2.
			elif self.road_length < self.dist < self.road_length + 0.25*np.pi*self.lane_width:
				self.angle = 3.*np.pi/2. - (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = -self.lane_width + 0.5*self.lane_width*np.cos(self.angle + np.pi/2.)	
				self.pos_y = self.lane_width + 0.5*self.lane_width*np.sin(self.angle + np.pi/2.)				
			elif self.road_length + 0.25*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.25*np.pi*self.lane_width:
				self.pos_x = -self.lane_width - self.dist + (self.road_length + 0.25*np.pi*self.lane_width)
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			else:
				self.completed_task = True
		elif self.start_pos == 'S' and self.end_pos == 'N':
			if self.dist < 2.*self.road_length + 2.*self.lane_width:				
				self.pos_x = 0.5*self.lane_width
				self.pos_y = -self.lane_width - self.road_length + self.dist
				self.angle = np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'S' and self.end_pos == 'S':			
			if self.dist <= self.road_length:
				self.pos_x = 0.5*self.lane_width
				self.pos_y = -self.lane_width - self.road_length + self.dist
				self.angle = np.pi/2.
			elif self.road_length < self.dist < self.road_length + 0.5*np.pi*self.lane_width:
				self.angle = np.pi/2. + (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = 0.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = -self.lane_width + 0.5*self.lane_width*np.sin(self.angle - np.pi/2.)
			elif self.road_length + 0.5*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.5*np.pi*self.lane_width:
				self.pos_x = -0.5*self.lane_width
				self.pos_y = -self.lane_width - self.dist + (self.road_length + 0.5*np.pi*self.lane_width)
				self.angle = 3.*np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'S' and self.end_pos == 'E':			
			if self.dist <= self.road_length:
				self.pos_x = 0.5*self.lane_width
				self.pos_y = -self.lane_width - self.road_length + self.dist
				self.angle = np.pi/2.
			elif self.road_length < self.dist < self.road_length + 0.25*np.pi*self.lane_width:
				self.angle = np.pi/2. - (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = self.lane_width + 0.5*self.lane_width*np.cos(self.angle + np.pi/2.)	
				self.pos_y = -self.lane_width + 0.5*self.lane_width*np.sin(self.angle + np.pi/2.)				
			elif self.road_length + 0.25*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.25*np.pi*self.lane_width:
				self.pos_x = self.lane_width + self.dist - (self.road_length + 0.25*np.pi*self.lane_width)
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			else:
				self.completed_task = True
		elif self.start_pos == 'S' and self.end_pos == 'W':			
			if self.dist <= self.road_length:
				self.pos_x = 0.5*self.lane_width
				self.pos_y = -self.lane_width - self.road_length + self.dist
				self.angle = np.pi/2.
			elif self.road_length < self.dist < self.road_length + 0.75*np.pi*self.lane_width:
				self.angle = np.pi/2. + (self.dist - self.road_length) / (1.5*self.lane_width)
				self.pos_x = -self.lane_width + 1.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = -self.lane_width + 1.5*self.lane_width*np.sin(self.angle - np.pi/2.)				
			elif self.road_length + 0.75*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.75*np.pi*self.lane_width:
				self.pos_x = -self.lane_width - self.dist + (self.road_length + 0.75*np.pi*self.lane_width)
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			else:
				self.completed_task = True
		elif self.start_pos == 'E' and self.end_pos == 'N':			
			if self.dist <= self.road_length:
				self.pos_x = self.lane_width + self.road_length - self.dist
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			elif self.road_length < self.dist < self.road_length + 0.25*np.pi*self.lane_width:
				self.angle = np.pi/2. - (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = self.lane_width + 0.5*self.lane_width*np.cos(self.angle + np.pi/2.)	
				self.pos_y = self.lane_width + 0.5*self.lane_width*np.sin(self.angle + np.pi/2.)				
			elif self.road_length + 0.25*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.25*np.pi*self.lane_width:
				self.pos_x = 0.5*self.lane_width
				self.pos_y = self.lane_width + self.dist - (self.road_length + 0.5*np.pi*self.lane_width)
				self.angle = np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'E' and self.end_pos == 'S':			
			if self.dist <= self.road_length:
				self.pos_x = self.lane_width + self.road_length - self.dist
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			elif self.road_length < self.dist < self.road_length + 0.75*np.pi*self.lane_width:
				self.angle = np.pi + (self.dist - self.road_length) / (1.5*self.lane_width)
				self.pos_x = self.lane_width + 1.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = -self.lane_width + 1.5*self.lane_width*np.sin(self.angle - np.pi/2.)				
			elif self.road_length + 0.75*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.75*np.pi*self.lane_width:
				self.pos_x = -0.5*self.lane_width
				self.pos_y = -self.lane_width - self.dist + (self.road_length + 0.75*np.pi*self.lane_width)
				self.angle = 3.*np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'E' and self.end_pos == 'E':
			if self.dist <= self.road_length:
				self.pos_x = self.lane_width + self.road_length - self.dist
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			elif self.road_length < self.dist < self.road_length + 0.5*np.pi*self.lane_width:
				self.angle = np.pi + (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = self.lane_width + 0.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = 0.5*self.lane_width*np.sin(self.angle - np.pi/2.)
			elif self.road_length + 0.5*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.5*np.pi*self.lane_width:
				self.pos_x = self.lane_width + self.dist - (self.road_length + 0.5*np.pi*self.lane_width)
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			else:
				self.completed_task = True
		elif self.start_pos == 'E' and self.end_pos == 'W':
			if self.dist < 2.*self.road_length + 2.*self.lane_width:				
				self.pos_x = self.lane_width + self.road_length - self.dist
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			else:
				self.completed_task = True			
		elif self.start_pos == 'W' and self.end_pos == 'N':			
			if self.dist <= self.road_length:
				self.pos_x = -self.lane_width - self.road_length + self.dist
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			elif self.road_length < self.dist < self.road_length + 0.75*np.pi*self.lane_width:
				self.angle = 0. + (self.dist - self.road_length) / (1.5*self.lane_width)
				self.pos_x = -self.lane_width + 1.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = self.lane_width + 1.5*self.lane_width*np.sin(self.angle - np.pi/2.)				
			elif self.road_length + 0.75*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.75*np.pi*self.lane_width:
				self.pos_x = 0.5*self.lane_width
				self.pos_y = self.lane_width + self.dist - (self.road_length + 0.75*np.pi*self.lane_width)
				self.angle = np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'W' and self.end_pos == 'S':			
			if self.dist <= self.road_length:
				self.pos_x = -self.lane_width - self.road_length + self.dist
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			elif self.road_length < self.dist < self.road_length + 0.25*np.pi*self.lane_width:
				self.angle = np.pi/2. - (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = -self.lane_width + 0.5*self.lane_width*np.cos(self.angle + np.pi/2.)	
				self.pos_y = -self.lane_width + 0.5*self.lane_width*np.sin(self.angle + np.pi/2.)				
			elif self.road_length + 0.25*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.25*np.pi*self.lane_width:
				self.pos_x = -0.5*self.lane_width
				self.pos_y = -self.lane_width - self.dist + (self.road_length + 0.25*np.pi*self.lane_width)
				self.angle = 3.*np.pi/2.
			else:
				self.completed_task = True
		elif self.start_pos == 'W' and self.end_pos == 'E':
			if self.dist < 2.*self.road_length + 2.*self.lane_width:				
				self.pos_x = -self.lane_width - self.road_length + self.dist
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			else:
				self.completed_task = True	
		elif self.start_pos == 'W' and self.end_pos == 'W':
			if self.dist <= self.road_length:
				self.pos_x = -self.lane_width - self.road_length + self.dist
				self.pos_y = -0.5*self.lane_width
				self.angle = 0.
			elif self.road_length < self.dist < self.road_length + 0.5*np.pi*self.lane_width:
				self.angle = 0. + (self.dist - self.road_length) / (0.5*self.lane_width)
				self.pos_x = -self.lane_width + 0.5*self.lane_width*np.cos(self.angle - np.pi/2.)	
				self.pos_y = 0.5*self.lane_width*np.sin(self.angle - np.pi/2.)
			elif self.road_length + 0.5*np.pi*self.lane_width <= self.dist < 2.*self.road_length + 0.5*np.pi*self.lane_width:
				self.pos_x = -self.lane_width - self.dist + (self.road_length + 0.5*np.pi*self.lane_width)
				self.pos_y = 0.5*self.lane_width
				self.angle = np.pi
			else:
				self.completed_task = True
		return

	def update(self, acceleration=0., time=0.1):
		acceleration = min(max(acceleration, -30.), 10.)
		self.acceleration = acceleration
		self.speed += acceleration*time
		self.speed = min(max(self.speed, 0.), 30.)
		self.dist += self.speed*time + 0.5*acceleration*time**2.
		self.update_pos_angle()
		self.vel_x = self.speed*np.cos(self.angle)
		self.vel_y = self.speed*np.sin(self.angle)
		self.acc_x = self.acceleration*np.cos(self.angle)
		self.acc_y = self.acceleration*np.sin(self.angle)		
		return

class Intersection:
	def __init__(self, lane_width=10., road_length=100., step_time=0.1, cars_per_state=3, max_timesteps=Config.max_timesteps):
		self.actions_to_accelerations = np.asarray([-30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
		self.num_actions = self.actions_to_accelerations.shape[0]
		self.tot_reward = 0.
		self.lane_width = lane_width
		self.road_length = road_length
		self.step_time = step_time
		self.num_timesteps = 0
		self.max_timesteps = max_timesteps
		self.end_state = False
		self.directions = ['N', 'S', 'E', 'W']
		self.directions_to_end_pos = {'N': (0.5*lane_width, 
			lane_width+road_length), 'S': (-0.5*lane_width, 
			-lane_width-road_length), 'E': (lane_width+road_length, 
			-0.5*lane_width), 'W': (-lane_width-road_length,
			0.5*lane_width)}

		self.cars_per_state = cars_per_state
		self.state_size = (self.cars_per_state+1)*6+2
		self.directions_to_cars = defaultdict(list)
		self.cars = []
		self.active_car_indices = []
		self.num_completed = 0
		for i in range(len(self.directions)):
			car_i = Car(self.directions[i], np.random.choice(self.directions), 30., lane_width, road_length)
			self.directions_to_cars[self.directions[i]].append(car_i)
			self.cars.append(car_i)
			self.active_car_indices.append(i)
		self.states = self.get_states()

	# state for car i is pos_x,pos_y,vel_x,vel_y,acc_x,acc_y
	# and same for self.cars_per_state closest cars
	# and end_pos_x, end_pos_y
	def get_states(self):
		active_car_states = []
		for i in self.active_car_indices:
			active_car_states.append(np.zeros(((self.cars_per_state+1)*6+2)) + np.inf)
			car_i = self.cars[i]
			active_car_states[-1][0:6] = [car_i.pos_x, car_i.pos_y, car_i.vel_x, car_i.vel_y, car_i.acc_x, car_i.acc_y]
			closest_distances = []
			for j in self.active_car_indices:
				if i != j:
					car_j = self.cars[j]
					dist = (car_j.pos_x - car_i.pos_x)**2. + (car_j.pos_y - car_i.pos_y)**2.
					closest_distances.append((dist, j))
			closest_distances.sort()
			for k in range(self.cars_per_state):
				j = closest_distances[k][1]
				car_j = self.cars[j]
				active_car_states[-1][6*(k+1):6*(k+2)] = [car_j.pos_x, car_j.pos_y, car_j.vel_x, car_j.vel_y, car_j.acc_x, car_j.acc_y]
			end_pos = self.directions_to_end_pos[car_i.end_pos]
			active_car_states[-1][-1] = end_pos[-1]
			active_car_states[-1][-2] = end_pos[-2]
		return np.asarray(active_car_states, dtype=np.float32)

	def apply_action(self, actions):
		self.num_timesteps += 1
		# rewards = np.zeros(actions.shape[0]) - 0.1/actions.shape[0]
		rewards = np.zeros(actions.shape[0])
		for k in range(actions.shape[0]):
			i = self.active_car_indices[k]
			car_i = self.cars[i]
			car_i.update(self.actions_to_accelerations[actions[k]], self.step_time)
		for k in range(actions.shape[0]):
			i = self.active_car_indices[k]
			car_i = self.cars[i]
			for l in range(k+1, actions.shape[0]):					
				j = self.active_car_indices[l]
				car_j = self.cars[j]
				dist = (car_j.pos_x - car_i.pos_x)**2. + (car_j.pos_y - car_i.pos_y)**2.
				if dist <= 8.**2.:
					self.end_state = True
					rewards[k] = -1000.
					rewards[l] = -1000.
					self.tot_reward += rewards.sum()
					return rewards, None
		for k in range(actions.shape[0]):
			i = self.active_car_indices[k]
			car_i = self.cars[i]
			if car_i.completed_task:
				self.num_completed += 1
				rewards[k] += 100.
				del self.active_car_indices[k]
		self.add_new_cars()
		self.states = self.get_states()
		self.tot_reward += rewards.sum()
		return rewards, self.states

	def add_new_cars(self):
		for direction in self.directions:
			car_dir = self.directions_to_cars[direction][-1]
			if car_dir.dist > 30.:				
				car_new = Car(direction, np.random.choice(self.directions), 30., self.lane_width, self.road_length)
				self.directions_to_cars[direction].append(car_new)
				self.active_car_indices.append(len(self.cars))
				self.cars.append(car_new)
		return 

	def is_end(self):
		return self.end_state or self.num_timesteps >= self.max_timesteps



class QN:
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self):
    	self.inters = Intersection()
        self.build()

    def add_placeholders_op(self):
        state_size = self.inters.state_size
        self.s = tf.placeholder(tf.float32, (None, state_size))
        self.a = tf.placeholder(tf.int32, (None))
        self.r = tf.placeholder(tf.float32, (None))



    def linear_get_q_values_op(self, state, scope, reuse=False):
        num_actions = self.inters.num_actions
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
    	num_actions = self.inters.num_actions
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
        num_actions = self.inters.num_actions

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
        random_actions = np.random.randint(0, self.inters.num_actions, states.shape[0])
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
	        self.inters = Intersection()
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
