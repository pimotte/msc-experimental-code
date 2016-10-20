#!/usr/bin/python2

DIR = "plots_walk/"

RECOMPUTE = True


import mcrep
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import datetime
import pickle

adaptor = mcrep.adaptor.Adaptor('multichain_exp.csv')

graph = adaptor.create_interaction_graph()
ordered_graph = adaptor.create_ordered_interaction_graph()

nodes = graph.nodes()

num_agents = len(nodes)

scores = {}

count = 0

for node in nodes:
    personal_nodes = [node1 for node1 in ordered_graph.nodes() if node1[0] == node]
    personal_nodes_numbers = [node1[1] for node1 in personal_nodes]
    personalisation = {(node, min(personal_nodes_numbers)): 1}

    #print ((node, -1) in ordered_graph.nodes())
    #print ([node1 for node1 in ordered_graph.nodes() if node1[0] == node])
    pimrank = mcrep.pimrank.PimRank(ordered_graph, personalisation).compute()

    number_of_nodes = len(personal_nodes)
    personalisation = dict(zip(personal_nodes,[1.0/number_of_nodes]*number_of_nodes))

    pimrank_spread = mcrep.pimrank.PimRank(ordered_graph, personalisation).compute()

    personalisation = {node : 1}
    pagerank = mcrep.pimrank.PimRank(graph, personalisation, weight='capacity').compute(False)

    scores[node] = (pimrank, pimrank_spread, pagerank) 

    count += 1 

    print "Computed " + str(count) + " out of " + str(num_agents)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


with open('pr_scores', 'w') as f:
    pickle.dump(scores, f)


