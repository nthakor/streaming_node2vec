import networkx as nx
import numpy as np
import random
# G=nx.read_edgelist('graph/karate.edgelist',nodetype=int)
G=nx.karate_club_graph()
K=nx.Graph()
sample_len=len(G.edges())
while(len(K.edges())<sample_len):
    n1=random.choice(G.nodes())
    n2=random.choice(G.nodes())
    e=(n1,n2)
    if(e not in G.edges()):
        K.add_edge(*e)
i=0
for x in G.edges():
    if(x in K.edges()):
        i+=1
print i
