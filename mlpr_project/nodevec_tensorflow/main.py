from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import networkx as nx
import node2vec
from itertools import chain
import collections
import math
import os
import random
from six.moves import urllib
import tensorflow as tf
data_index = 0

def main():

	walk_length=10
	num_walks=20
	p=1
	q=1
	# nx_G = read_graph()
	nx_G = nx.read_edgelist('karate.edgelist', nodetype=int, create_using=nx.DiGraph())
	for edge in nx_G.edges():
		nx_G[edge[0]][edge[1]]['weight'] = 1
	G = node2vec.Graph(nx_G, False, p, q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(num_walks, walk_length)
	walks = [map(str, walk) for walk in walks]
	words=list(chain.from_iterable(walks))

	print('Data size', len(words))

	# Step 2: Build the dictionary and replace rare words with UNK token.
	vocabulary_size = 34
	
	def build_dataset(words):
	  count = [['UNK', -1]]
	  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	  dictionary = dict()
	  for word, _ in count:
	    dictionary[word] = len(dictionary)
	  data = list()
	  unk_count = 0
	  for word in words:
	    if word in dictionary:
	      index = dictionary[word]
	    else:
	      index = 0  
	      unk_count += 1
	    data.append(index)
	  count[0][1] = unk_count
	  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	  return data, count, dictionary, reverse_dictionary
	
	data, count, dictionary, reverse_dictionary = build_dataset(words)
	del words  
	print('Most common words (+UNK)', count[:5])
	print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
	
	
	
	
	def generate_batch(batch_size, num_skips, skip_window):
	  global data_index
	  assert batch_size % num_skips == 0
	  assert num_skips <= 2 * skip_window
	  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
	  buffer = collections.deque(maxlen=span)
	  for _ in range(span):
	    buffer.append(data[data_index])
	    data_index = (data_index + 1) % len(data)
	  for i in range(batch_size // num_skips):
	    target = skip_window  # target label at the center of the buffer
	    targets_to_avoid = [ skip_window ]
	    for j in range(num_skips):
	      while target in targets_to_avoid:
	        target = random.randint(0, span - 1)
	      targets_to_avoid.append(target)
	      batch[i * num_skips + j] = buffer[skip_window]
	      labels[i * num_skips + j, 0] = buffer[target]
	    buffer.append(data[data_index])
	    data_index = (data_index + 1) % len(data)
	  return batch, labels
	
	batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
	for i in range(8):
	  print(batch[i], reverse_dictionary[batch[i]],
	      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
	
	
	batch_size = 20
	embedding_size = 128  # Dimension of the embedding vector.
	skip_window = 1       # How many words to consider left and right.
	num_skips = 2         # How many times to reuse an input to generate a label.
	
	valid_size = 5     # Random set of words to evaluate similarity on.
	valid_window = 20  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)
	num_sampled = 20    # Number of negative examples to sample.
	
	graph = tf.Graph()
	
	with graph.as_default():
	
	  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	  with tf.device('/cpu:0'):
	    embeddings = tf.Variable(
	        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
	
	    nce_weights = tf.Variable(
	        tf.truncated_normal([vocabulary_size, embedding_size],
	                            stddev=1.0 / math.sqrt(embedding_size)))
	    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
	
	  loss = tf.reduce_mean(
	      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
	                     num_sampled, vocabulary_size))
	
	  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
	
	  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	  normalized_embeddings = embeddings / norm
	  valid_embeddings = tf.nn.embedding_lookup(
	      normalized_embeddings, valid_dataset)
	  similarity = tf.matmul(
	      valid_embeddings, normalized_embeddings, transpose_b=True)
	
	  init = tf.initialize_all_variables()
	
	num_steps = 100001
	
	with tf.Session(graph=graph) as session:
	  init.run()
	  print("Initialized")
	
	  average_loss = 0
	  for step in xrange(num_steps):
	    batch_inputs, batch_labels = generate_batch(
	        batch_size, num_skips, skip_window)
	    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
	
	    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
	    average_loss += loss_val
	
	    if step % 2000 == 0:
	      if step > 0:
	        average_loss /= 2000
	      print("Average loss at step ", step, ": ", average_loss)
	      average_loss = 0
	
	  final_embeddings = normalized_embeddings.eval()





if __name__ == "__main__":
	main()
