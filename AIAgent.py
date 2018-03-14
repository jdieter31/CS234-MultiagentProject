
# coding: utf-8

# In[1]:

from DQN import QN
from multiprocessing.dummy import Pool as ThreadPool 
import tensorflow as tf
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import traceback
import time
import sys


# In[2]:

def create_player(player,rows,columns):
	try:
		model = QN(player=player,rows=rows,columns=columns)
		print('starting player',player)
		model.run()
	except Exception as e:
		print(e)    
		traceback.print_exc()


# In[3]:

team=1
n_players=int(sys.argv[1])
rows=int(sys.argv[2])
columns=int(sys.argv[3])


# In[5]:

p = ThreadPool(n_players) 
executor = ThreadPoolExecutor(max_workers=n_players)


# In[6]:

for i in range(1,n_players+1):
	g = tf.Graph()
	with g.as_default():
		executor.submit(create_player,i,rows,columns)
	time.sleep(1)


# In[7]:

executor.shutdown()


# In[8]:



