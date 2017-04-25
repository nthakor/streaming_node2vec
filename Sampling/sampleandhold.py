import networkx as nx
import csv
import numpy as np
r1index = 0
r2index = 0

p = 0.1
q = 0.7

G = nx.Graph()
no_of_edges=0
limit=10000
i=0
with open('karate.edgelist', 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
		if(i>limit):
			break
		i+=1
		n1 = row[0]
		n2 = row[1]
		add_indicator = 0;
		if(len(G.nodes())==0):
			G.add_edge(n1,n2)
		else:
			for node in G.nodes():
				if(node==n1 or node==n2):
					r=np.random.uniform()
					if(r<=p):
						G.add_edge(n1,n2)
						add_indicator=1
						break
		if(add_indicator==0):
			r=np.random.uniform()
			if(r<=q):
				G.add_edge(n1,n2)

print nx.edges(G),len(G.edges())
print "|G|: %d |E|: %d"%(len(G.nodes()),len(G.edges()))