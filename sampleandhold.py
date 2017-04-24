import networkx as nx
import csv
import random

r1index = 0
r2index = 0

p = 0
q = 0

G = nx.Graph()
no_of_edges=0

with open('karate.edgelist', 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
		node1 = row[0]
		node2 = row[1]
		# print node1,node2
		# print nx.number_of_nodes(G)
		add_indicator = 0;
		for i in range(0,max(nx.number_of_nodes(G),1)):
			if (nx.number_of_nodes(G)==0):
				G.add_edge(node1, node2)
				add_indicator = 1
				break
			elif (node1 == G.nodes()[i] or node2 == G.nodes()[i]): 
				r1 = random.uniform(0,1)
				if (r1 <= p):
					# r1index+=1
					G.add_edge(node1, node2)
					add_indicator = 1
				break
		if (add_indicator==0):
			r2 = random.uniform(0,1)
			if (r2 <= q):
				# r2index+=1
				G.add_edge(node1, node2)

		no_of_edges+=1

print nx.edges(G),len(G.edges())
# print r1index, r2index
