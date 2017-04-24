
# coding: utf-8

# In[57]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import genfromtxt
import itertools


# In[38]:

df=pd.read_csv('emb/karate.emb',skiprows=1,sep=' ',header=None)
df=df.sort_values(by=[0],ascending=True)
emb=df.as_matrix().astype(np.float32)
emb=emb[:,1:]


# In[39]:

G=nx.read_edgelist('graph/karate.edgelist',nodetype=int)
edgeLen=len(G.edges())
edges_G=G.edges()
edges_G=np.asarray(edges_G,dtype=int)
np.random.shuffle(edges_G)


# In[40]:

edge_features_1=np.empty(shape=[0,emb.shape[1]])
for edge in edges_G:
    node1=emb[edge[0]-1]
    node2=emb[edge[1]-1]
    avg=np.mean([node1,node2],axis=0)
    hadamard=np.multiply(node1,node2)
    diff=node1-node2
    wL1=np.absolute(diff)
    WL2=diff**2
    feature=hadamard
    edge_features_1=np.vstack([edge_features_1,hadamard])


# In[41]:

G_clique=nx.Graph()
nodeL=G.nodes()
for [node1,node2] in itertools.product(nodeL,nodeL):
    if(node1!=node2):
        G_clique.add_edge(node1,node2)
G_clique.remove_edges_from(G.edges())
edges_G_clique=G_clique.edges()
edges_G_clique=np.asarray(edges_G_clique,dtype=int)
randIdx=np.random.randint(0,len(edges_G_clique),len(edges_G))
edges_G_clique=edges_G_clique[randIdx]


# In[42]:

edge_features_0=np.empty(shape=[0,emb.shape[1]])
for edge in edges_G_clique:
    node1=emb[edge[0]-1]
    node2=emb[edge[1]-1]
    avg=np.mean([node1,node2],axis=0)
    hadamard=np.multiply(node1,node2)
    diff=node1-node2
    wL1=np.absolute(diff)
    WL2=diff**2
    feature=hadamard
    edge_features_0=np.vstack([edge_features_0,hadamard])


# In[43]:

class_0=np.zeros([edgeLen,1])
class_1=np.ones([edgeLen,1])
edge_features_0=np.hstack([edge_features_0,class_0])
edge_features_1=np.hstack([edge_features_1,class_1])
dataset=np.vstack([edge_features_0,edge_features_1])


# In[44]:

dataset.shape


# In[55]:

for i in range(10):
    np.random.shuffle(dataset)
X=dataset[:,:dataset.shape[1]-1]
y=dataset[:,dataset.shape[1]-1:]
y=np.reshape(y,[y.shape[0],])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[62]:

# from sklearn import svm
# clf = svm.SVC(kernel='rbf')
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)


# In[63]:

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# In[65]:

from sklearn.metrics import f1_score,precision_score,recall_score
f1=f1_score(y_test, y_pred, average='macro')
precision=precision_score(y_test, y_pred, average='macro')
recall=recall_score(y_test, y_pred, average='macro')


# In[66]:

f1, precision, recall

