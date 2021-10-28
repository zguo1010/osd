# script to do a random walk on a graph, starting from a random node of a given side and counting the number of *times* we end up in either a node from the same side or from the other side
# cite from https://github.com/gvrkiran/controversy-detection/blob/master/code/randomwalk/computePolarizationScoreRandomwalk.py
# random walk type3.

import networkx as nx
import numpy as np
import random, sys
from operator import itemgetter

def getRandomNodes(G, k):  # parameter k = number of random nodes to generate
    nodes = G.nodes()
    random_nodes = {}
    for i in range(k):
        random_num = random.randint(0, len(nodes) - 1)
        random_nodes[nodes[random_num]] = 1
    return random_nodes


def getRandomNodesFromLabels(G, k, flag, left, right):
    # parameter k = no. of random nodes to generate, flag could be "left", "right" or "both". If both, k/2 from one
    # side and k/2 from the other side are generated.
    random_nodes = []
    random_nodes1 = {}
    if flag == "left":
        for i in range(k):
            random_num = random.randint(0, len(left) - 1)
            random_nodes.append(left[random_num])
    elif flag == "right":
        for i in range(k):
            random_num = random.randint(0, len(right) - 1)
            random_nodes.append(right[random_num])
    else:
        for i in range(k / 2):
            random_num = random.randint(0, len(left) - 1)
            random_nodes.append(left[random_num])
        for i in range(k / 2):
            random_num = random.randint(0, len(right) - 1)
            random_nodes.append(right[random_num])
    for ele in random_nodes:
        random_nodes1[ele] = 1
    return random_nodes1


def performRandomWalk(G, starting_node, user_nodes_side1, user_nodes_side2):
    # returns if we ended up in a "left" node or a "right" node;
    dict_nodes = {}  # contains unique nodes seen till now;
    nodes = G.nodes()
    num_edges = len(G.edges())
    step_count = 0
    #	total_other_nodes = len(user_nodes.keys())
    flag = 0
    side = ""

    while flag != 1:
        # print "starting from ", starting_node, "num nodes visited ", len(dict_nodes.keys()), " out of ", len(nodes)
        neighbors = list(G.neighbors(starting_node))
        if len(neighbors) <= 1: ###in case of empty range
            return "left"
        random_num = random.randint(0, len(neighbors) - 1)
        starting_node = neighbors[random_num]
        dict_nodes[starting_node] = 1
        step_count += 1
        if starting_node in user_nodes_side1:
            side = "left"
            flag = 1
        if starting_node in user_nodes_side2:
            side = "right"
            flag = 1
    #		if(step_count>num_edges**2): # if stuck
    #			break
    #		if(step_count%100000==0):
    #			print >> sys.stderr, step_count, "steps reached"
    return side


def getDict(nodes_list):
    dict_nodes = {}
    for node in nodes_list:
        dict_nodes[node] = 1
    return dict_nodes



