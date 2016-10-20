#!/usr/bin/python2

DIR = "plots/"


import mcrep
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import datetime
import time
import pickle
import random
import os.path

random.seed(0)

adaptor = mcrep.adaptor.Adaptor('multichain_exp.csv')
#adaptor = mcrep.adaptor.Adaptor('test.csv')

count = 0

scores = {}

times = []
key_times = {}

for line in adaptor.iterate_csv(None):
    time_str = line[-1].strip()
    pubkey1 = line[0]
    pubkey2 = line[1]
    insert_time = time.mktime(time.strptime(time_str, '"%Y-%m-%d %H:%M:%S"'))

    key_times[pubkey1] = min(key_times.get(pubkey1, time.time()), insert_time)
    key_times[pubkey2] = min(key_times.get(pubkey2, time.time()), insert_time)


    times.append(insert_time)

times.sort()



time_representants = []

for i in xrange(1, 10):
    time_representants.append(times[int(i*len(times)/11.0)])

key_times = sorted(key_times.values())
for i in xrange(1, 10):
    time_representants.append(key_times[int(i*len(key_times)/11.0)])

time_representants.sort()

print(time_representants)

node = None

compute_times = []

for time_rep in time_representants:


    pre_time = time.time()
    graph = adaptor.create_interaction_graph(filter_date=time_rep)
    post_build = time.time()

    if node == None:
        node = random.choice(graph.nodes())
        print(node)



    algo = mcrep.netflow.Netflow(graph, identity=node, alpha=2.)
    algo.compute()
    post_time = time.time()
    
    time_netflow_build = post_build - pre_time
    time_netflow_compute = post_time - post_build


    pre_time = time.time()
    ordered_graph = adaptor.create_ordered_interaction_graph(time_rep)

    personal_nodes = [node1 for node1 in ordered_graph.nodes() if node1[0] == node]
    number_of_nodes = len(personal_nodes)
    personalisation = dict(zip(personal_nodes,[1.0/number_of_nodes]*number_of_nodes))

    post_build = time.time()
    pimrank_spread = mcrep.pimrank.PimRank(ordered_graph, personalisation).compute()
    post_time = time.time()
    
    time_pimrank_build = post_build - pre_time
    time_pimrank_compute = post_time - post_build

    data = (len(graph.nodes()), len(ordered_graph.nodes()), time_rep, time_netflow_build, time_netflow_compute, time_pimrank_build, time_pimrank_compute)

    print(data)

    compute_times.append(data)
            
with open('results/performance', 'w') as f:
    pickle.dump(compute_times, f)


