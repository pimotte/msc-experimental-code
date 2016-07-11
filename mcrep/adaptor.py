#!/usr/bin/env python2

import networkx as nx

class Adaptor:
    """
    The Adaptor class serves as a way to create networkx DiGraph objects
    from a csv
    """

    def __init__(self, csvfile):
        self.csvfile = csvfile


    def create_interaction_graph(self):
        G = nx.DiGraph()
        
        for interaction in self.iterate_csv():
            info = interaction[:4]

            pubkey_req = info[0]
            pubkey_res = info[1]
            up = int(info[2])
            down = int(info[3])

            self.update_graph(G, pubkey_req, pubkey_res, up)
            self.update_graph(G, pubkey_res, pubkey_req, down)
        return G



    def update_graph(self, graph, pubkey1, pubkey2, contrib):
        try:
            old = graph[pubkey1][pubkey2]['capacity']
            graph.add_edge(pubkey1, pubkey2, capacity=(old+contrib))
        except KeyError:
            graph.add_edge(pubkey1, pubkey2, capacity=contrib)


    def create_ordered_interaction_graph(self):
        G = nx.DiGraph()

        for interaction in self.iterate_csv():
            pubkey_requester = interaction[0]
            pubkey_responder = interaction[1]

            sequence_number_requester = int(interaction[6])
            sequence_number_responder = int(interaction[12])
            contribution_requester = int(interaction[2])
            contribution_responder = int(interaction[3])

            G.add_edge((pubkey_requester, sequence_number_requester), (pubkey_requester, sequence_number_requester+1),
                    contribution=contribution_requester)
            G.add_edge((pubkey_responder, sequence_number_responder), (pubkey_responder, sequence_number_responder+1),
                    contribution=contribution_responder)

        return G

    def iterate_csv(self):
        with open(self.csvfile, 'r') as f:
            for line in f:
                # Format:
                # 0: Pubkey requester
                # 1: Pubkey responder
                # 2: Up (data requester -> responder)
                # 3: Down (data responder -> requester)
                # 4: Total_Up_Requester
                # 5: Total_Down_Requester
                # 6: Sequence_Number_Requester
                # 7: Previous_Hash_Requester
                # 8: Signature_Requester
                # 9: Hash_Responder
                # 10: Total_Up_Responder
                # 11: Total_Down_Responder
                # 12: Sequence_Number_Responder
                # 13: Previous_Hash_Responder
                # 14: Signature_Responder
                # 15: Hash_Responder
                # 16: Insert_Time

                if line.startswith("Public"):
                    continue
                else:
                    yield line.split(";")
         



