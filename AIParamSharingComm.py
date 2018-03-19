
# coding: utf-8

# In[1]:

from DQN_Param_Share_Com_Agent import QNParamSharingComAgent
from DQN_Param_Share_Com_Learner import QNParamSharingComLeaner
from multiprocessing.dummy import Pool as ThreadPool 
import tensorflow as tf
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import traceback
import time
import sys
#reload('DQN_Param_Share_Agent')
#reload('DQN_Param_Share_Learner')

# In[2]:

# In[2]:

def create_player(player,rows,columns):
    try:
        model = QNParamSharingComAgent(player=player,rows=rows,columns=columns)
        print('starting player',player)
        model.run()
    except Exception as e:
        print(e)    
        traceback.print_exc()

def create_learner(player,rows,columns):
    try:
        g = tf.Graph()
        with g.as_default():
            model = QNParamSharingComLeaner(player=player,rows=rows,columns=columns)
            print('starting learner')
            model.run()
    except Exception as e:
        print(e)    
        traceback.print_exc()


# In[3]:

print('why!??')

team=1
n_players=int(sys.argv[1])
rows=int(sys.argv[2])
columns=int(sys.argv[3])



# In[4]:

p = ThreadPool(n_players) 
executor = ThreadPoolExecutor(max_workers=n_players + 1)


# In[5]:

g = tf.Graph()
with g.as_default():
    executor.submit(create_learner,-1,rows,columns)


# In[6]:

for i in range((team-1)*n_players,team*n_players):
    g = tf.Graph()
    with g.as_default():
        executor.submit(create_player,i,rows,columns)
    time.sleep(1)



# In[ ]:

executor.shutdown()

