import networkx as nx
import csv
import numpy as np
import random
sample_size = 20
G = nx.Graph()
no_of_edges=0

with open('karate.edgelist', 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
		node1 = row[0]
		node2 = row[1]
		if (len(G.edges()) < sample_size):
			G.add_edge(node1, node2)
		else:
			p_e = float(float(len(G.edges()))/float(no_of_edges))		
			r = np.random.uniform()
			if (r <= p_e):
				rand_edge = random.randint(0, sample_size - 1)
				G.remove_edge(G.edges()[rand_edge][0],G.edges()[rand_edge][1])
				G.add_edge(node1, node2)
		no_of_edges+=1

print nx.edges(G),len(G.edges())

