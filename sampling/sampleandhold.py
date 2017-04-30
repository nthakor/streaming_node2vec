import networkx as nx
import csv,argparse
import numpy as np


parser = argparse.ArgumentParser(description='Graph Priority Sampling by node degree')
parser.add_argument('-input', '--input_file', help='Input file name', required=True)
parser.add_argument('-p', '--p', help='Similarity Index', required=True,type=float)
parser.add_argument('-q', '--q', help='Dissimilarity Index', required=True,type=float)
arguments = vars(parser.parse_args())

r1index = 0
r2index = 0
p = arguments['p']
q = arguments['q']
# q = 0.9

G = nx.Graph()
no_of_edges=0
i=0

with open(arguments['input_file'], 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=' ')
	for row in f: 
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
		no_of_edges += 1
		print no_of_edges

print nx.edges(G),len(G.edges())
print "|G|: %d |E|: %d"%(len(G.nodes()),len(G.edges()))

nx.write_edgelist(G,'{}.1_sampleandhold'.format(arguments['input_file']),delimiter=' ',data=False)