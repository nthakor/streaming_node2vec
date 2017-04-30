import networkx as nx
import csv,random,argparse
from heapq import heappush, heappop, nsmallest


parser = argparse.ArgumentParser(description='Graph Priority Sampling by counting triangles')
parser.add_argument('-input', '--input_file', help='Input file name', required=True)
parser.add_argument('-r', '--sample_ratio', help='Sampling Percentage', required=True,type=int)
arguments = vars(parser.parse_args())

K=nx.read_edgelist(arguments['input_file'],nodetype=int)
sample_size=int(arguments['sample_ratio']*len(K.edges())/100)

G = nx.Graph()
G1 = nx.Graph()
no_of_edges=0 
h=[]
count = sample_size/10

with open(arguments['input_file'], 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
		node1 = row[0]
		node2 = row[1]
		min_degree = 1

		if (G.has_node(node1) == True and G.has_node(node2) == True):
			min_degree = min(G.degree(node1),G.degree(node2))
			if(min_degree == 0):
				min_degree = 1
		elif(G.has_node(node1) == True and G.has_node(node2) == False):
			min_degree = G.degree(node1)
			if(min_degree == 0):
				min_degree = 1
		elif(G.has_node(node1) == False and G.has_node(node2) == True):
			min_degree = G.degree(node2)
			if(min_degree == 0):
				min_degree = 1
		else:
			pass

		u=0
		w = float(1/float(min_degree))
		print min_degree,w,no_of_edges 
		# print node1, node2, w
		while u==0:
			u = random.uniform(0,1)
		r = float(w/u)
		# print r

		if (nx.number_of_edges(G) < sample_size):
			G.add_edge(node1, node2)
			# print node1,node2
			heappush(h,(r,node1,node2))
		elif (r>nsmallest(1,h)[0][0]):
			pop = heappop(h)
			# print pop[1],pop[2]
			if (G.has_edge(pop[1],pop[2])==True):
				G.remove_edge(pop[1],pop[2])
			else:
				G.remove_edge(pop[2],pop[1])
			heappush(h,(r,node1,node2))
			G.add_edge(node1, node2)
			# print node1,node2
		else:
			pass

		count = count-1
		no_of_edges+=1

# print len(G.edges())
print len(G.edges()),nx.edges(G)
# nx.write_edgelist(G1,'{}.1_gpsnodes_sample'.format(arguments['input_file']),delimiter=' ',data=False)
nx.write_edgelist(G,'{}.{}_gpsnodes_sample'.format(arguments['input_file'],arguments['sample_ratio']),delimiter=' ',data=False)
