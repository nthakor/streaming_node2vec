import networkx as nx
import csv,random,argparse
import numpy as np
sample_size = 20
G = nx.Graph()
no_of_edges=0

parser = argparse.ArgumentParser(description='Simulate a cache')
parser.add_argument('-input', '--input_file', help='Input file name', required=True)
parser.add_argument('-r', '--sample_ratio', help='Sampling Percentage', required=True,type=int)
arguments = vars(parser.parse_args())
K=nx.read_edgelist(arguments['input_file'],nodetype=int)
sample_size=int(arguments['sample_ratio']*len(K.edges())/100)

with open(arguments['input_file'], 'rb') as csvfile:
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
		print no_of_edges

print nx.edges(G),len(G.edges())
nx.write_edgelist(G,'{}.{}_reservoir_sample'.format(arguments['input_file'],arguments['sample_ratio']),delimiter=' ',data=False)
