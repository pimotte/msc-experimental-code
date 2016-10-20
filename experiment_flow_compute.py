#!/usr/bin/python2

DIR = "plots/"


import mcrep
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import datetime
import pickle
import random
import os.path

adaptor = mcrep.adaptor.Adaptor('multichain_exp.csv')
#adaptor = mcrep.adaptor.Adaptor('test.csv')

graph = adaptor.create_interaction_graph()
print("Graph contains: " + str(len(graph.nodes())) + " nodes")
ordered_graph = adaptor.create_ordered_interaction_graph()


nodes = graph.nodes()

num_agents = len(nodes)

count = 0

scores = {}


for node in nodes:
    fname = 'results/flow_scores_' + str(node)
    if (os.path.isfile(fname)):
        continue

    algo = mcrep.netflow.Netflow(graph, identity=node, alpha=1.)

    algo.compute()
    
    count += 1 

    print "Computed " + str(count) + " out of " + str(num_agents*len(alphas))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    score_graph = algo.graph

    
    with open('results/flow_scores_' + str(node), 'w') as f:
        pickle.dump(score_graph, f)


