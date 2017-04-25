
import networkx as nx
import csv
import random
from heapq import heappush, heappop, nsmallest

sample_size = 30

G = nx.Graph()
no_of_edges=0
h=[]

with open('karate.edgelist', 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
		node1 = row[0]
		node2 = row[1]
		total_triangles=0
		total_triangles_temp=0
		u=0
		
		total_triangles=sum(nx.triangles(G,G.nodes()).values())/3
		G.add_edge(node1,node2) 
		total_triangles_temp=sum(nx.triangles(G,G.nodes()).values())/3

		w = total_triangles_temp - total_triangles
		# print total_triangles,total_triangles_temp,w,no_of_edges 
		print node1, node2, w
		while u==0:
			u = random.uniform(0,1)
		r = float(w/u)
		print r
		G.remove_edge(node1,node2)

		if (nx.number_of_edges(G) < sample_size):
			G.add_edge(node1, node2)
			heappush(h,(r,node1,node2))

		elif (r>nsmallest(1,h)[0][0]):
			pop = heappop(h)
			G.remove_edge(pop[1],pop[2])
			heappush(h,(r,node1,node2))
			G.add_edge(node1, node2)

		no_of_edges+=1

print nx.edges(G),len(G.edges())

