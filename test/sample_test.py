import networkx as nx
import numpy as np
import random
from sklearn.neighbors import KDTree
G=nx.read_edgelist('graph/karate.edgelist',nodetype=int)
# G=nx.karate_club_graph()
K=nx.Graph()
sample_len=len(G.edges())
E=np.asarray(G.edges(),dtype=int)
print E.shape
tree=KDTree(E)
while(len(K.edges())<sample_len):
    n1=random.choice(G.nodes())
    n2=random.choice(G.nodes())
    a=np.empty(shape=(0,2))
    print a.shape
    a=np.vstack((a,np.array([n1,n2])))
    dist,ind = tree.query(a[0],k=1)
    if(int(dist)!=0):
    	K.add_edge(n1,n2)

i=0
for x in G.edges():
    if(x in K.edges()):
        i+=1
print i
