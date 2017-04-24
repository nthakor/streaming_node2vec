import networkx as nx
import csv
import random

sample_size = 20

G = nx.Graph()
no_of_edges=0
# f = open("fb.txt","rb")


with open('karate.edgelist', 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
		node1 = row[0]
		node2 = row[1]
		# print node1,node2
		if (nx.number_of_edges(G) < sample_size):
			G.add_edge(node1, node2)
		else:
			p_e = len(G)/no_of_edges		
			r = random.random()
			if (r <= p_e):
				rand_edge = random.randint(0, sample_size - 1)
				G.remove_edge(G.edges()[rand_edge][0],G.edges()[rand_edge][1])
				G.add_edge(node1, node2)
		no_of_edges+=1

print nx.edges(G),len(G.edges())

