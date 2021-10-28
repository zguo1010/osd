#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_m
import warnings
import math
import time
import random
import json
from numpy.random import randint, choice, seed
from datetime import date
import sys
from multiprocessing import Process, Value, Array, Pool
from game_functions_nash import *
plt.switch_backend('agg')

def main(argv): #100, 0.1, 0.1, 0, 0.8, 0.9, 20
    #parameters
    dname = "Cresci15"
    N = 1000
    T = int(argv[0]) #1000
    pa = float(argv[1]) #0.1
    pt = float(argv[2]) #0.1
    puh = float(argv[3]) #0.0 for H, 1.0 for U
    mup, sigmap = float(argv[4]), 0.1 #0.6, 0.1
    mur, sigmar = float(argv[5]), 0.05 #0.7, 0.05
    run = int(argv[6])
    method = 'consensus'

    print("N: %d T: %d pa: %f pt: %f puh: %f run: %d" %(N, T, pa, pt, puh, run))
    #strategy choices for attackers, defenders, and users
    #choices: A-k, D-l, U-m
    strategy = {"A": ['DG', 'C', 'DN', 'S'], "D":['T', 'M'],
                "U":['SU', 'U', 'NU'], "H":['SU', 'U', 'NU'], "T":['SU', 'U', 'NU']}
    #cost of defender's strategy
    defender_cost = {'T':0.1, 'M':0}

    #read data from files
    likers = pd.read_csv("cresci_combined_TWT.csv") # post=tweet, share=retweets, comment=reply, like=favorite
    print(likers.columns) #'num_hashtags' replaces 'category' as support received from friends/followers
    print(likers.shape)
    likers = pd.read_csv("cresci_combined_TWT.csv",
                         names=['user_id', 'followers', 'friends', 'age', 'lines',
                               'len_name', 'ave_share', 'ave_comment', 'ave_like', 'category', #'num_hashtags'
                               'num_urls', 'num_mentions', 'favorite_tweets', 'total_posts',
                               'total_replies', 'freq_posts', 'freq_replies', 'label', 'net_posts', 'freq_np', 'verified'],
                         header=0)
    full_size = likers.shape[0]
    likers['total_posts'] = likers['total_posts'] - likers['total_replies']
    topics = pd.read_csv("TopicsSpamTFPE13.csv", names=range(20), header=None)

    #set max friend
    likers.loc[likers['friends']>8000, 'friends'] = 8000
    likers.loc[likers['friends']==0, 'friends'] = 1

    # reduce friend number by scale
    scale = 0.08
    likers['friends'] = np.ceil(likers['friends'] * scale)

    # normalize large values
    likers.loc[likers['favorite_tweets']>300, 'favorite_tweets'] = 300
    maximum_cap(likers, 'freq_posts', 0.1)
    maximum_cap(likers, 'freq_replies', 0.1)

    # features P^f P^p and friends
    ld = likers['label'] == 'Legit'
    likers.loc[ld, 'feeding'] = (likers.loc[ld, 'freq_replies'] / max(likers.loc[ld, 'freq_replies'])
                                 + likers.loc[ld, 'favorite_tweets'] / max(likers.loc[ld, 'favorite_tweets'])) / 2
    likers.loc[ld, 'posting'] = likers.loc[ld, 'freq_posts'] / max(likers.loc[ld, 'freq_posts'])
    likers.loc[ld, 'inviting'] = likers.loc[ld, 'friends'] / np.percentile(likers.loc[ld, 'friends'], 92)
    print('friend max: ', max(likers['friends']))
    f = int(np.percentile(likers['friends'], 90))
    print('friend 90%:', f)



    # start game decision
    start_time_i = time.time()
    opinions_runs = {x:{} for x in range(T)} #save each user's opinions for runs/interactions
    between_runs = {x:{} for x in range(T)} #save betweenness metric for runs
    redundancy_runs = {x:{} for x in range(T)} #save redundancy metric for runs
    opinions_std, between_std, redundancy_std = {x:{} for x in range(T)}, {x:{} for x in range(T)}, {x:{} for x in range(T)}
    components = {x:{} for x in range(run)}
    components_st = {x:{} for x in range(run)}
    choice_at, choice_df, choice_u, choice_h = np.empty([0,4]),  np.empty([0,2]), np.empty([0,3]),  np.empty([0,3])
    utility_at, utility_df, utility_u, utility_h = np.empty([0, 4]), np.empty([0, 2]), np.empty([0, 3]), np.empty([0, 3])
    nodes = {'T':[], 'A':[], 'U':[], 'H':[]} #save the user types
    nodes_total = []

    G, topics_s, nodes, nodes_total = initialization(likers, topics, N, pa, pt, puh, mup, sigmap, mur, sigmar)
    # make friend connection
    friendship_TP(G, dname, topics_s)  # same friend work
    # if r == 0:
    print(G.degree())
    print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    #save betweeness and redundancy to file
    capital_between(G)
    capital_redundancy(G)
    between_0 = list(G.nodes(data='capital'))
    redundancy_0 = list(G.nodes(data='redundancy'))

    #degree plot
    if puh == 1.0:
        friends = list(G.degree())
        friends_value = [y for (x, y) in friends]
        plt.hist(friends_value, bins=20, color = 'black')
        plt.xlabel('degree of nodes', fontsize=16)
        plt.ylabel('frequency', fontsize=16)
        plt.savefig('results/degree_c_'+ str(int(puh*100)) +'.png')
        plt.clf()

    # parallel running
    directory = 'rescre/tmp%d/' % int(puh * 100)
    files_in_directory = os.listdir(directory)
    filtered_files = sorted([file for file in files_in_directory if file.endswith(".txt")])
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)

    threads = [0 for r in range(run)]
    for r in range(run):
        threads[r] = Process(target=game_run, args=(G, dname, N, T, pa, puh, r, defender_cost, method))
        threads[r].start()
    for thread in threads:
        thread.join()
    start_time_r = time.time()
    print("running time for parallel", run, (start_time_r - start_time_i) / 60)

    #collect all results in opinions_runs
    directory = 'rescre/tmp%d/' % int(puh * 100)
    files_in_directory = os.listdir(directory)
    filtered_files = sorted([file for file in files_in_directory if file.endswith(".txt")])
    for filename in filtered_files:
        file = open(directory + filename, 'r')
        print("open a new file", filename)
        counter = 0
        #read first interaction
        line = file.readline()
        components_st[int(filename.split('.')[0])] = [int(x) for x in line[1:-2].split(', ')]
        # read T interaction
        line = file.readline()
        components[int(filename.split('.')[0])] = [int(x) for x in line[1:-2].split(', ')]
        line = file.readline()
        while counter <= T:
            if line.startswith("INTER "):
                t = int(line.split(' ')[1])
                counter += 1
            else:
                linesplit = line.split(' [')
                node = int(linesplit[0])
                new_op = np.array([float(x) for x in linesplit[1][:-2].split(', ')]).reshape((1,4))
                if node not in opinions_runs[t]:
                    opinions_runs[t][node] = new_op
                else:
                    opinions_runs[t][node] = np.append(opinions_runs[t][node], new_op, axis=0)
            line = file.readline()
        counter = 1
        while counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            else:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in between_runs[t]:
                    between_runs[t][node] = [float(linesplit[1])]
                else:
                    between_runs[t][node].append(float(linesplit[1]))
            line = file.readline()
        counter = 1
        while line and counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            else:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in redundancy_runs[t]:
                    redundancy_runs[t][node] = [float(linesplit[1])]
                else:
                    redundancy_runs[t][node].append(float(linesplit[1]))
            line = file.readline()
        if sum([float(x) for x in line[1:-2].split(', ')]) > 0:
            choice_at = np.append(choice_at, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 4)), axis=0)
        line = file.readline()
        choice_df = np.append(choice_df, np.array([int(x) for x in line[1:-2].split(', ')]).reshape((1, 2)), axis=0)
        line = file.readline()
        if puh != 0.0:
            choice_u = np.append(choice_u, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 3)), axis=0)
        line = file.readline()
        if puh != 1.0:
            choice_h = np.append(choice_h, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 3)), axis=0)
        # utility
        line = file.readline()
        if sum([float(x) for x in line[1:-2].split(', ')]) != 0:
            utility_at = np.append(utility_at, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 4)), axis=0)
            print("###", utility_at)
        line = file.readline()
        utility_df = np.append(utility_df, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 2)), axis=0)
        line = file.readline()
        if puh != 0.0:
            utility_u = np.append(utility_u, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 3)), axis=0)
        line = file.readline()
        if puh != 1.0:
            utility_h = np.append(utility_h, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 3)), axis=0)
        file.close()
        print("file", filename, "closed")
    print(components)
    print("running time data collection", (time.time() - start_time_r)/60)

    #result for runs and interactions
    for t in range(T):
        for n in nodes_total:
            if n in opinions_runs[t]:
                opinions_std[t][n] = np.std(opinions_runs[t][n], axis = 0)
                opinions_runs[t][n] = np.mean(opinions_runs[t][n], axis = 0)
                between_std[t][n] = np.std(np.array(between_runs[t][n]))
                between_runs[t][n] = np.mean(np.array(between_runs[t][n]))
                redundancy_std[t][n] = np.std(np.array(redundancy_runs[t][n]))
                redundancy_runs[t][n] = np.mean(np.array(redundancy_runs[t][n]))
    #calculate strategy choices ratio
    print(choice_at)
    print(choice_df)
    print(choice_u)
    print(choice_h)
    print(utility_at)
    print(utility_df)
    print(utility_u)
    print(utility_h)
    choice_at = choice_at / np.sum(choice_at, axis=1, keepdims=True)
    choice_df = choice_df / np.sum(choice_df, axis=1, keepdims=True)
    if puh != 0:
        choice_u = choice_u / np.sum(choice_u, axis=1, keepdims=True)
    if puh != 1:
        choice_h = choice_h / np.sum(choice_h, axis=1, keepdims=True)

    #save to file -- opinions_runs
    with open('node_c_'+ str(int(puh*100)) +'.txt', 'w') as f:
        #write parameters
        f.write("%s\n" % list(argv))
        f.write("%s\n" % list(nodes['A']))
        f.write("%s\n" % list(nodes['T']))
        f.write("%s\n" % list(nodes['U']))
        f.write("%s\n" % list(nodes['H']))
        f.write("%s\n" % list(np.mean(choice_at, axis = 0)))
        f.write("%s\n" % list(np.mean(choice_df, axis = 0)))
        if puh != 0:
            f.write("%s\n" % list(np.mean(choice_u, axis = 0)))
        else:
            f.write("%s\n" % list(choice_u))
        if puh != 1.0:
            f.write("%s\n" % list(np.mean(choice_h, axis = 0)))
        else:
            f.write("%s\n" % list(choice_h))
        f.write("%s\n" % list(np.mean(utility_at, axis = 0)))
        f.write("%s\n" % list(np.mean(utility_df, axis = 0)))
        if puh != 0:
            f.write("%s\n" % list(np.mean(utility_u, axis = 0)))
        else:
            f.write("%s\n" % list(utility_u))
        if puh != 1.0:
            f.write("%s\n" % list(np.mean(utility_h, axis = 0)))
        else:
            f.write("%s\n" % list(utility_h))

        for item in components: #run lines
            f.write("%s\n" % components_st[item])
        for item in components: #run lines
            f.write("%s\n" % components[item])
        # save betweeness and redundancy of the initial network
        for (x,y) in between_0: #N lines
            f.write("%d %f\n" % (x, y))
        for (x,y) in redundancy_0: #N lines
            f.write("%d %f\n" % (x, y))

    with open('op_c_'+ str(int(puh*100)) +'.txt', 'w') as f:
        for item in opinions_runs:
            f.write("INTER %s\n" % item)
            for node in opinions_runs[item]:
                f.write("%d %s\n" % (node, list(opinions_runs[item][node])))
        for item in opinions_std:
            f.write("INTER %s\n" % item)
            for node in opinions_std[item]:
                f.write("%d %s\n" % (node, list(opinions_std[item][node])))

        for item in between_runs:
            f.write("INTER %s\n" % item)
            for node in between_runs[item]:
                f.write("%d %f\n" % (node, between_runs[item][node]))
        for item in between_std:
            f.write("INTER %s\n" % item)
            for node in between_std[item]:
                f.write("%d %f\n" % (node, between_std[item][node]))

        for item in redundancy_runs:
            f.write("INTER %s\n" % item)
            for node in redundancy_runs[item]:
                f.write("%d %f\n" % (node, redundancy_runs[item][node]))
        for item in redundancy_std:
            f.write("INTER %s\n" % item)
            for node in redundancy_std[item]:
                f.write("%d %f\n" % (node, redundancy_std[item][node]))

    print("total time: ", (time.time() - start_time_i)/60)

if __name__ == '__main__':
    main(sys.argv[1:])