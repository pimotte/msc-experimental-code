#!/usr/bin/env python2

import networkx as nx

class Netflow:
    """
    This class implements the Netflow algorithm.
    """

    def __init__(self, graph, identity=None, alpha=2):
        self.graph = graph
        
        if identity is not None:
            self.identity = identity 
        else:
            self.identity = graph.nodes()[0]

        for neighbour in self.graph.out_edges([self.identity], 'capacity', 0):
            
            cap = self.graph.edge[self.identity][neighbour[1]]['capacity'] 
            self.graph.edge[self.identity][neighbour[1]]['capacity'] = float(cap)/float(alpha)

    def compute(self):
        self.initial_step()
        self.transform_graph()
        self.netflow_step()
    
    def initial_step(self):
        """
        In the intial step, all capactities are computed
        """

        for node in self.graph.nodes_iter():
            self.compute_capacity(node)

    def transform_graph(self):
        """
        In this step, the graph is transformed, based on the capacities computed in step 1
        """

        self.augmented_graph = nx.DiGraph()

        for edge in self.graph.edges_iter(data=True):
            self.augmented_graph.add_edge(edge[0] + "OUT", edge[1] + "IN", {'capacity' : edge[2]['capacity']})

        for node in self.graph.nodes_iter(data=True):
            if node[0] == self.identity:
                self.augmented_graph.add_edge(node[0] + "IN", node[0] + "OUT")
            else:
                self.augmented_graph.add_edge(node[0] + "IN", node[0] + "OUT", {'capacity' : node[1]['capacity']})


    def netflow_step(self):

        for node in self.graph.nodes_iter():
            if node == self.identity:
                self.graph.node[node]['score'] = 0

                continue
            score = nx.maximum_flow_value(self.augmented_graph, node + "IN", self.identity + "OUT")

            self.graph.node[node]['score'] = score


    def compute_capacity(self, node):
        if node == self.identity:
            return
        contribution = nx.maximum_flow_value(self.graph, node, self.identity)
        consumption = nx.maximum_flow_value(self.graph, self.identity, node)

        self.graph.add_node(node, capacity=max(0, contribution-consumption))


