#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx
import warnings
import math
import time
import random
from numpy.random import randint, choice, seed
from datetime import date
import sys, os
from multiprocessing import Process, Value, Array
from game_functions import *
plt.switch_backend('agg')

def hist_bt_rd_old(data_list, data_start, nodes_type, dname, feature, T, f, puh, run, model):
    directory = 'res1ks/tmp%s/' % model
    data_runs = {r:{} for r in range(run)}
    for r in range(run):
        fname = directory + str(r) + '.txt'
        file = open(fname, 'r')
        counter = 0
        file.readline()
        file.readline()
        line = file.readline()
        while counter <= T:
            if line.startswith("INTER "):
                counter += 1
            line = file.readline()
        counter = 1
        while counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            elif feature == 'bt' and counter == T:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in nodes_type['A'] and node not in nodes_type['T'] and float(linesplit[1]) >= 0.001:
                    if len(data_runs[r]) == 0:
                        data_runs[r] = [float(linesplit[1])]
                    else:
                        data_runs[r].append(float(linesplit[1]))
            line = file.readline()
        counter = 1
        while line and counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            elif feature == 'rd' and counter == T:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in nodes_type['A'] and node not in nodes_type['T']:
                    if len(data_runs[r]) == 0:
                        data_runs[r] = [float(linesplit[1])]
                    else:
                        data_runs[r].append(float(linesplit[1]))
            line = file.readline()
        file.close()

    print("plot "+feature)
    bt1 = data_start
    bt2 = data_list[T-1]
    bt1_value = [y for (x, y) in bt1.items() if (x not in nodes_type['A'] and x not in nodes_type['T'])]
    bt2_value = [y for (x, y) in bt2.items() if (x not in nodes_type['A'] and x not in nodes_type['T'])]
    if feature == 'bt':
        bt1_value = [y for (x, y) in bt1.items() if (x not in nodes_type['A'] and x not in nodes_type['T'] and y >= 0.001)]
        bt2_value = [y for (x, y) in bt2.items() if (x not in nodes_type['A'] and x not in nodes_type['T'] and y >= 0.001)]

    xmin = min(min(bt1_value), min(bt2_value))
    xmax = max(max(bt1_value), max(bt2_value)) #set to specific value
    result1 = np.histogram(bt1_value, bins=20, range=(xmin, xmax))
    x1 = result1[1] #s
    result2 = np.histogram(bt2_value, bins=20, range=(xmin, xmax))
    x2 = result2[1] #s
    width = result1[1][1] - result1[1][0]
    height = max(max(result1[0]),(max(result2[0]))) #set to specific value
    # height = max(result1[0])
    hist_list = np.zeros((run, 20))
    for r in data_runs:
        bt = data_runs[r]
        hist_list[r] = np.histogram(bt, bins=20, range=(xmin, xmax))[0]
    x2_std = np.std(hist_list, axis=0)

    if feature == 'bt':
        colors = ['orange', 'b']
    else:
        colors = ['g', 'r']
    # Draw histogram with density for Fake class
    plt.figure(figsize=(6, 3.6))
    plt.gcf().subplots_adjust(top=0.98)
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(right=0.98)
    if feature == 'bt':
        height = 200
    elif feature == 'rd':
        height = 600
    plt.ylim(0, height+20)
    plt.bar(x1[:-1], result1[0], width=width * 0.75, color=colors[0], alpha=0.8, label='before') #/cnt1
    plt.bar(x2[:-1], result2[0], yerr=x2_std, width=width * 0.75, color=colors[1], alpha=0.5, label='after') #/cnt2

    if feature == 'bt':
        plt.xlabel('betweenness metric', fontsize=24)
    else:
        plt.xlabel('redundancy metric', fontsize=24)
    plt.ylabel('frequency', fontsize=24)
    # plt.title('sc by redundancy', fontsize=16)
    plt.tick_params(axis='both', labelsize=18)
    if feature == 'bt':
        plt.xticks(np.arange(0, 0.02, step=0.005))
    elif feature == 'rd':
        plt.xticks(np.arange(0, 1, step=0.2))
    plt.legend(loc='upper right', markerscale=4, fontsize=16)
    plt.savefig('results/%sh_%s_%s.png' % (feature, dname, model))
    plt.clf()

def hist_bt_rd(data_list, data_start, nodes_type, dname, feature, T, f, puh, run, model):
    directory = 'res1ks/tmp%s/' % model
    if f == 'c':
        directory = 'rescre/tmp%s/' % model
    data_runs = {r:{} for r in range(run)}
    files_in_directory = os.listdir(directory)
    for r in range(run):
        fname = directory + str(r) + '.txt'
        if fname not in files_in_directory:
            continue
        file = open(fname, 'r')
        counter = 0
        file.readline()
        file.readline()
        line = file.readline()
        while counter <= T:
            if line.startswith("INTER "):
                counter += 1
            line = file.readline()
        counter = 1
        while counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            elif feature == 'bt' and counter == T:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in nodes_type['A'] and node not in nodes_type['T'] and float(linesplit[1]) >= 0.001:
                    if len(data_runs[r]) == 0:
                        data_runs[r] = [float(linesplit[1])]
                    else:
                        data_runs[r].append(float(linesplit[1]))
            line = file.readline()
        counter = 1
        while line and counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            elif feature == 'rd' and counter == T:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in nodes_type['A'] and node not in nodes_type['T']:
                    if len(data_runs[r]) == 0:
                        data_runs[r] = [float(linesplit[1])]
                    else:
                        data_runs[r].append(float(linesplit[1]))
            line = file.readline()
        file.close()

    print("plot "+feature)
    bt1 = data_start
    bt2 = data_list[T-1]
    bt1_value = [y for (x, y) in bt1.items() if (x not in nodes_type['A'] and x not in nodes_type['T'])]
    bt2_value = [y for (x, y) in bt2.items() if (x not in nodes_type['A'] and x not in nodes_type['T'])]
    if feature == 'bt':
        bt1_value = [y for (x, y) in bt1.items() if (x not in nodes_type['A'] and x not in nodes_type['T'] and y > 0.001)] #0.001
        bt2_value = [y for (x, y) in bt2.items() if (x not in nodes_type['A'] and x not in nodes_type['T'] and y > 0.001)] #0.001
    print("sum of before: ", sum(bt1.values()), "after: ", sum(bt2.values()))
    print("ave of before: ", sum(bt1.values())/len(bt1), "after: ", sum(bt2.values())/len(bt2))
    xmin = min(min(bt1_value), min(bt2_value))
    xmax = max(max(bt1_value), max(bt2_value)) #set to specific value
    result1 = np.histogram(bt1_value, bins=20, range=(xmin, xmax))
    x1 = result1[1] #s
    result2 = np.histogram(bt2_value, bins=20, range=(xmin, xmax))
    x2 = result2[1] #s
    width = result1[1][1] - result1[1][0]
    height = max(max(result1[0]),(max(result2[0]))) #set to specific value
    # height = max(result1[0])
    # hist_list = np.zeros((run, 20))
    hist_list = np.empty([0, 20])
    for r in data_runs:
        fname = directory + str(r) + '.txt'
        if fname not in files_in_directory:
            continue
        bt = data_runs[r]
        # hist_list[r] = np.histogram(bt, bins=20, range=(xmin, xmax))[0]
        hist_list.append(hist_list, np.histogram(bt, bins=20, range=(xmin, xmax))[0], axis=0)
    x2_std = np.std(hist_list, axis=0)

    if feature == 'bt':
        colors = ['orange', 'b']
    else:
        colors = ['g', 'r']
    # Draw histogram with density for Fake class
    plt.figure(figsize=(6, 3.1))
    if feature == 'bt':
        plt.gcf().subplots_adjust(top=0.98)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(right=0.98)
        # plt.ylim(0, math.log10(600+20))
        plt.yscale('log')
        plt.plot(0, 400, color='w')
        if f == 'k':
            plt.plot(0.1, 600, color='w')
    elif feature == 'rd':
        plt.gcf().subplots_adjust(top=0.97)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(right=0.98)
        height = 250
        if f == 'c':
            height = 330
        plt.ylim(0, height)
    plt.bar(x1[:-1], result1[0], width=width * 0.75, color=colors[0], alpha=0.8, label='before') #/cnt1
    plt.bar(x2[:-1], result2[0], yerr=x2_std, width=width * 0.75, color=colors[1], alpha=0.5, label='after') #/cnt2

    if feature == 'bt':
        plt.xlabel('betweenness metric', fontsize=22)
    else:
        plt.xlabel('trust score', fontsize=22)
        if f == 'k':
           plt.xticks(np.arange(0, 1, step=0.2))
    plt.ylabel('frequency', fontsize=24)
    # plt.title('sc by redundancy', fontsize=16)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(loc='upper right', markerscale=4, fontsize=16)
    plt.savefig('results/%sh_%s_%s.png' % (feature, dname, model))
    plt.clf()

def plot_opinions(opinions_runs, nodes_type, T, f, puh, model, hist = True):
    '''Time-series plot of opinions'''
    print("plot opinions")
    plt.figure(figsize=(5.4, 4.2))
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(right=0.99)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.gcf().subplots_adjust(top=1)
    colors = ['blue', 'red', 'green']
    u = 0
    for x in range(T):
        for node in opinions_runs[x]:
            # if node in nodes_type[user]:
            if node not in nodes_type['A'] and node not in nodes_type['T']:
                belief = opinions_runs[x][node][0]
                disbelief = opinions_runs[x][node][1]
                uncertainty = opinions_runs[x][node][2]
                if u == 0:
                    plt.plot(x + 1, belief, 'o', color=colors[0], markersize=1, label='b')
                    plt.plot(x + 1, disbelief, 'o', color=colors[1], markersize=1, label='d')
                    plt.plot(x + 1, uncertainty, 'o', color=colors[2], markersize=1, label='u')
                    u += 1
                else:
                    plt.plot(x + 1, uncertainty, 'o', color=colors[2], markersize=1)
                    plt.plot(x + 1, disbelief, 'o', color=colors[1], markersize=1)
                    plt.plot(x + 1, belief, 'o', color=colors[0], markersize=1)

    plt.xlabel('number of interactions per node', fontsize=22, x=0.45)
    plt.ylabel('b, d, or u', fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    plt.savefig('results/op_%s_%s.png' % (f, model))
    # plt.show()
    plt.clf()

    if hist:
        # print("plot opinions", user)
        plt.figure(figsize=(6, 3.6))
        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(right=0.96)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(top=1.0)
        h_b, h_d, h_u, h_a = [], [], [], []
        for i in opinions_runs[T-1]:
            # if i in nodes_type[user]:
            if i not in nodes_type['A'] and i not in nodes_type['T']:
                tmp = opinions_runs[T-1][i][:]
                h_b.append(tmp[0])
                h_d.append(tmp[1])
                h_u.append(tmp[2])
                h_a.append(tmp[3])
        plt.hist(sorted(h_b), bins=50, color = colors[0])
        plt.xlabel('belief b', fontsize=24)
        plt.ylabel('number of users', fontsize=24)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig('results/oph_%s_%s_b.png' % (f, model))
        plt.clf()

        plt.hist(sorted(h_d), bins=50, color = colors[1])
        plt.xlabel('disbelief d', fontsize=24)
        plt.ylabel('number of users', fontsize=24)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig('results/oph_%s_%s_d.png' % (f, model))
        plt.clf()

        plt.hist(sorted(h_u), bins=50, color = colors[2])
        plt.xlabel('uncertainty u', fontsize=24)
        plt.ylabel('number of users', fontsize=24)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig('results/oph_%s_%s_u.png' % (f, model))
        plt.clf()

        plt.hist(sorted(h_a), bins=50, color = 'gray')
        plt.xlabel('base rate a', fontsize=24)
        plt.ylabel('number of users', fontsize=24)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig('results/oph_%s_%s_a.png' % (f, model))
        plt.clf()

def main(argv): #4_29, k, 0, d
    dt = argv[0]
    f = argv[1] #"c"/"k"
    puh = int(argv[2])
    N = 1000
    model = argv[3]
    print("dt: %s f: %s puh: %d " %(dt, f, puh))

    # N = 1000
    # run = 1
    # pa = 0.1
    # pt = 0.1
    # # puh in [0, 25, 50, 63, 75, 100]

    start_time = time.time()
    # Read parameters and nodes
    nodes_type = {'A':[], 'T':[], 'U':[], 'H':[]} #save the user types

    file = open('results/%s/node_%s_%s.txt' %(dt,f,model), 'r')
    # file = open('node_%s_%s.txt' % (f, model), 'r') #this location has an error of much longer time
    line = file.readline()
    paras = [x for x in line[1:-2].split(', ')]
    print(paras)
    T, pa, pt, run = int(paras[0].strip("'")), float(paras[1].strip("'")), float(paras[2].strip("'")), int(paras[6].strip("'"))
    print("T: %d pa: %f pt: %f run:%d" % (T, pa, pt, run))
    line = file.readline()
    nodes_type['A'] = [int(x) for x in line[1:-2].split(', ')] #', ')
    line = file.readline()
    nodes_type['T'] = [int(x) for x in line[1:-2].split(', ')]
    line = file.readline()
    if puh != 0:
        nodes_type['U'] = [int(x) for x in line[1:-2].split(', ')]
    line = file.readline()
    if puh != 100:
        nodes_type['H'] = [int(x) for x in line[1:-2].split(', ')]
    #strategy choices
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    #utility
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    #read component list after 1 and all interactions
    for r in range(run): #components list
        file.readline()
    for r in range(run): #components list
        file.readline()
    #read redundancy and betweenness before interactions
    between_start = {}
    redundancy_start = {}
    for node in range(N):
        line = file.readline()
        between_start[int(line[:-1].split()[0])] = float(line[:-1].split()[1])
    for node in range(N):
        line = file.readline()
        redundancy_start[int(line[:-1].split()[0])] = float(line[:-1].split()[1])
    file.close()
    print(between_start)
    print(redundancy_start)

    # Read opinions
    opinions_runs, between_runs, redundancy_runs = {x1:{} for x1 in range(T)}, {x:{} for x in range(T)} , {x:{} for x in range(T)}
    opinions_std, between_std, redundancy_std = {x:{} for x in range(T)}, {x:{} for x in range(T)}, {x:{} for x in range(T)}

    file = open('results/%s/op_%s_%s.txt' % (dt, f, model), 'r')
    # file = open('op_%s_%s.txt' % (f, model), 'r')
    counter = 0
    line = file.readline()
    while counter <= T:
        if line.startswith("INTER "):
            t = int(line.split(' ')[1])
            counter += 1
        else:
            linesplit = line.split(' [')
            node = int(linesplit[0])
            opinions_runs[t][node] = np.array([float(x) for x in linesplit[1][:-2].split(', ')])
        line = file.readline()
    counter = 1
    while counter <= T:
        if line.startswith("INTER "):
            t = int(line.split(' ')[1])
            counter += 1
        else:
            linesplit = line.split(' [')
            node = int(linesplit[0])
            new_op = np.array([float(x) for x in linesplit[1][:-2].split(', ')]).reshape((1, 4))
            opinions_std[t][node] = new_op
        line = file.readline()

    # Read betweenness and redundancy
    counter = 1
    while counter <= T:
        if line.startswith("INTER "):
            t = int(line[:-1].split(' ')[1])
            counter += 1
        else:
            linesplit = line.split(' ')
            node = int(linesplit[0])
            between_runs[t][node] = float(linesplit[1])
        line = file.readline()
    counter = 1
    while counter <= T:
        if line.startswith("INTER "):
            t = int(line[:-1].split(' ')[1])
            counter += 1
        else:
            linesplit = line.split(' ')
            node = int(linesplit[0])
            between_std[t][node] = float(linesplit[1])
        line = file.readline()

    counter = 1
    while line and counter <= T:
        if line.startswith("INTER "):
            t = int(line[:-1].split(' ')[1])
            counter += 1
        else:
            linesplit = line.split(' ')
            node = int(linesplit[0])
            redundancy_runs[t][node] = float(linesplit[1])
        line = file.readline()
    counter = 1
    while line and counter <= T:
        if line.startswith("INTER "):
            t = int(line[:-1].split(' ')[1])
            counter += 1
        else:
            linesplit = line.split(' ')
            node = int(linesplit[0])
            redundancy_std[t][node] = float(linesplit[1])
        line = file.readline()
    file.close()
    del file
    print('total reading time: ', (time.time()-start_time)/60)

    # linear version for bt plots
    start_time = time.time()
    hist_bt_rd(between_runs,    between_start,    nodes_type, f, 'bt', T, f, puh, run, model)
    hist_bt_rd(redundancy_runs, redundancy_start, nodes_type, f, 'rd', T, f, puh, run, model)
    print('total time: ', (time.time()-start_time)/60)

if __name__ == '__main__':
    main(sys.argv[1:])

