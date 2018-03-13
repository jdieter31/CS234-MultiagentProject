
# coding: utf-8

# In[1]:

import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
import sys


# In[2]:

hand_coded_path='../GridSoccerSimulator/Bin/HandCodedClient.exe'
team=2
n_players=int(sys.argv[1])
rows=20
columns=30


# In[3]:

def create_player(player,rows,columns):
    try:
        print('starting player',player)
        subprocess.check_output(['mono',hand_coded_path,'-name','HandCoded','-n',str(player)])
    except Exception as e:
        print(e)    


# In[4]:

executor = ThreadPoolExecutor(max_workers=n_players)


# In[5]:

for i in range(1, n_players+1):
    executor.submit(create_player,i,rows,columns)
    time.sleep(1)


# In[ ]:



