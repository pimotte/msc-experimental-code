#!/usr/bin/python2

DIR = "plots_all/"



import mcrep
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import pickle
import time

with open('results/performance', 'r') as f:
    times = pickle.load(f)

number_of_nodes = np.fromiter((time[0] for time in times), np.float)
number_of_interactions = np.fromiter((time[1] for time in times), np.float)
moments_in_time = np.fromiter((time[2] for time in times), np.float)
time_netflow_build = np.fromiter((time[3] for time in times), np.float)
time_netflow_compute = np.fromiter((time[4] for time in times), np.float)
time_netflow_total = np.fromiter((time[3] + time[4] for time in times), np.float)
time_pimrank_build = np.fromiter((time[5] for time in times), np.float)
time_pimrank_compute = np.fromiter((time[6] for time in times), np.float)
time_pimrank_total = np.fromiter((time[5] + time[6] for time in times), np.float)


plt.plot(number_of_nodes, time_netflow_total, label="Total Time")
plt.plot(number_of_nodes, time_netflow_build, label="Preparation Time")
plt.plot(number_of_nodes, time_netflow_compute, label="Computation Time")
plt.title('Time to compute net-flow')
plt.xlabel('Number of agents')
plt.ylabel('Time (seconds)')
plt.legend(loc='upper left')
plt.savefig(DIR + 'plot_time_netflow.svg')
plt.clf()

plt.plot(number_of_interactions, time_netflow_total, label="Total Time")
plt.plot(number_of_interactions, time_netflow_build, label="Preparation Time")
plt.plot(number_of_interactions, time_netflow_compute, label="Computation Time")
plt.title('Time to compute net-flow')
plt.xlabel('Number of interactions')
plt.ylabel('Time (seconds)')
plt.legend(loc='upper left')
plt.savefig(DIR + 'plot_time_interactions_netflow.svg')
plt.clf()

plt.plot(number_of_nodes, time_pimrank_total, label="Total Time")
plt.plot(number_of_nodes, time_pimrank_build, label="Preparation Time")
plt.plot(number_of_nodes, time_pimrank_compute, label="Computation Time")

plt.title('Time to compute Temporal PageRank')
plt.xlabel('Number of nodes')
plt.ylabel('Time (seconds)')
plt.legend(loc='upper left')
plt.savefig(DIR + 'plot_time_nodes_pimrank.svg')
plt.clf()

plt.plot(number_of_interactions, time_pimrank_total, label="Total Time")
plt.plot(number_of_interactions, time_pimrank_build, label="Preparation Time")
plt.plot(number_of_interactions, time_pimrank_compute, label="Computation Time")

plt.title('Time to compute Temporal PageRank')
plt.xlabel('Number of interactions')
plt.ylabel('Time (seconds)')
plt.legend(loc='upper left')
plt.savefig(DIR + 'plot_time_pimrank.svg')
plt.clf()

plt.plot(moments_in_time, number_of_nodes)

plt.title('Number of agents in network over time')
plt.xlabel('Time (unix timestamp)')
plt.ylabel('Number of agents')
plt.savefig(DIR + 'plot_time_agents.svg')
plt.clf()


plt.plot(moments_in_time, number_of_interactions)

plt.title('Number of agents in network over time')
plt.xlabel('Time (unix timestamp)')
plt.ylabel('Number of interactions')
plt.savefig(DIR + 'plot_time_interactions.svg')
plt.clf()

