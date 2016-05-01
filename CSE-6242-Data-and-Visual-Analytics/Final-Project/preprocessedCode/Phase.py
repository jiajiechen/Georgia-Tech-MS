
# coding: utf-8

# In[1]:

# Find Nodes' each phase

import pandas as pd
import sqlite3
import numpy as np
#ht = pd.read_csv('./GTMMC_CSVs/Tweet_Hashtag.csv')
# Create your connection. 
conn = sqlite3.connect('GTMMC.sqlite3')
conn.text_factory = str
ht = pd.read_sql_query("SELECT * FROM Tweet_Hashtag", conn) 
tweet = pd.read_sql_query("SELECT * FROM Tweet_Data", conn)
links = pd.read_sql_query("SELECT * FROM User_Tweet_Link", conn)
ht['HashText'] = ht['HashText'].str.lower()
rank = pd.read_csv('./GTMMC_CSVs/inferencerRank_t6.csv')

m = pd.read_sql_query("SELECT t.Creator_ID, ht.HashText, count(t.Tweet_ID) as freq FROM Tweet_Hashtag ht LEFT JOIN Tweet_Data t ON t.Tweet_ID = ht.Tweet_ID GROUP BY t.Creator_ID, ht.HashText ORDER BY t.Creator_ID, freq desc ", conn)


# In[31]:

links = pd.read_csv("./GTMMC_CSVs/linkData_new.csv")
infl = pd.read_csv("./GTMMC_CSVs/influencers.csv")


# In[44]:
#links[(links['source'] == 86369703) | (links['target'] == 86369703)]
# In[67]:

#links = pd.read_csv('./GTMMC_CSVs/User_Tweet_Link.csv')
# In[24]:
#tweet[tweet['Creator_ID']=='3313699878']
# In[51]:
#edge.sort(['phase', 'source_weekend_home'], ascending=[True, False])['source'].unique()
# In[236]:

edge = pd.read_csv("./Edge/t5Edge.csv")
influ = pd.read_csv("./Edge/inferencerRank_t5.csv")
edge.head()
#influ.head()
#edge[edge['source']==96697085]
#t = edge.sort(['phase', 'source_weekend_home'], ascending=[True, False])['source'].unique()
#t1 = np.array(influ.iloc[:,:].userID) # 1, 2, 3, -5 influencers
# In[237]:
# topic-related threshold
# t1: 0.02; t2: 0.09; t4: 0.22; t5: 0; t6: 0; t7:0.03; t8:0.02; t10: 0
#ph1 = edge.sort(['phase', 'source_weekend_home'], ascending=[True, False])['source'].unique()[0:1]
ph1 = np.array(influ.iloc[0:1,:].userID) # 1, 2, 3, -5 influencers
t2 = edge[edge['source'].isin(ph1) * edge['phase'] == 2] 
ph2 = t2[t2['target_weekend_home'] > 0]['target'] # update for each topic: column name & threshold
t3 = edge[edge['source'].isin(ph2)* edge['phase'] == 3]
ph3 = t3[t3['target_weekend_home'] > 0]['target'] # update for each topic: column name & threshold
print len(ph1), len(ph2), len(ph3)

# append into phase data
a = np.unique(np.append(edge.source, edge.target))
node = pd.DataFrame({'UserID':a[:]})

node['PhaseI'] = 0
node.PhaseI[node['UserID'].isin(ph1)] = 1
node['PhaseII'] = 0
node.PhaseII[node['UserID'].isin(ph2)] = 1
node['PhaseIII'] = 0
node.PhaseIII[node['UserID'].isin(ph3)] = 1
print node.PhaseI.sum(), node.PhaseII.sum(), node.PhaseIII.sum()
node.to_csv('t5_Node1.csv') # change save file name


# In[ ]:



