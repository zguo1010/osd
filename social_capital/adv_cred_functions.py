#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import scipy.stats
import warnings
import math
import time
import random
from numpy.random import choice, randint, seed
from datetime import date
from copy import deepcopy
import os
from multiprocessing import Process, Value, Array

'''change max values'''
def maximum_cap(data, feature='ave_share', thresh=0.05, binnum=10):
    result1 = np.histogram(data.loc[data['label'] == "Fake", :][feature], bins=binnum)
    cnt1 = sum(data['label'] == "Fake")
    xmax1 = result1[1][-1]

    # Legit class
    result2 = np.histogram(data.loc[data['label'] == "Legit", :][feature], bins=binnum)
    cnt2 = sum(data['label'] == "Legit")
    xmax2 = result2[1][-1]

    # set x-axix cut
    xval = max(result1[1][sum(result1[0] >= cnt1 * thresh)],
               result2[1][sum(result2[0] >= cnt2 * thresh)])
    data.loc[data[feature] > xval, feature] = xval
    return

'''drawing best fit probability density functions'''
def fitting_distribution(data, feature='verified', dname='FakeLikers', method='SC', drawing=False, binnum=20, out=20):
    xmax = max(data[feature])
    xmin = min(data[feature])

    # Fake Class
    result1 = np.histogram(data.loc[data['label'] == "Fake", :][feature], bins=binnum, range=(xmin, xmax))
    cnt1 = sum(data['label'] == "Fake")
    x1 = result1[1]
    y1 = data.loc[data['label'] == "Fake", :][feature]
    # Legit class
    result2 = np.histogram(data.loc[data['label'] == "Legit", :][feature], bins=binnum, range=(xmin, xmax))
    cnt2 = sum(data['label'] == "Legit")
    x2 = result2[1]
    y2 = data.loc[data['label'] == "Legit", :][feature]

    # histogram with same height
    height = max(np.max(result1[0]) / cnt1, np.max(result2[0]) / cnt2)
    tmp = round(height * 10) / 10
    if tmp < height:
        height = tmp + 0.1
    else:
        height = tmp
    width = result1[1][1] - result1[1][0]
    dist_names = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'binom', 'bradford', 'burr', 'cauchy', 'chi',
                  'chi2', 'cosine',
                  'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                  'fisk',
                  'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto', 'gennorm', 'genexpon',
                  'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
                  'gumbel_l', 'halfcauchy', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss',
                  'halflogistic',
                  'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'levy_l',
                  # 'levy','levy_stable',
                  'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2',
                  'ncf',
                  'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal',
                  'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm',
                  'tukeylambda',
                  'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']
    # best fit probability model for Fake class
    best_name1 = 'expon'
    best_distribution1 = getattr(scipy.stats, best_name1)
    best_sse1 = np.inf
    best_params1 = (0.0, 1.0)
    for dist_name in dist_names:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Fit dist data
                dist = getattr(scipy.stats, dist_name)
                params = dist.fit(y1)
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(x1[:-1], loc=loc, scale=scale, *arg) * width
                sse = np.sum(np.power(result1[0] / cnt1 - pdf, 2.0))
                # Identify if this distribution is better
                if best_sse1 > sse > 0:
                    best_distribution1 = dist
                    best_params1 = params
                    best_sse1 = sse
                    best_name1 = dist_name
        except Exception:
            pass

    # plot the best fit curve for Fake class

    if drawing == True:
        plt.figure(figsize=(7, 5))
        plt.ylim(0, height)
        width = result1[1][1] - result1[1][0]
        if x1[-1] > 40:
            x1_list = np.arange(x1[-1])
        elif x1[-1] > 5:
            x1_list = np.arange(0, x1[-1], 0.2)
        elif x1[-1] > 0.5:
            x1_list = np.arange(0, x1[-1], 0.05)
        else:
            x1_list = np.arange(0, x1[-1], 0.005)
        pdf_fitted1 = best_distribution1.pdf(x1_list, *best_params1[:-2], loc=best_params1[-2],
                                             scale=best_params1[-1]) * width

        # Draw histogram with density for Fake class
        plt.bar(x1[:-1], result1[0] / cnt1, width=width * 0.75, color='r', alpha=0.1, label='Attackers')
        plt.plot(x1_list, pdf_fitted1, label='Attackers: ' + best_name1, color='r')

    # best fit probability model for Legit class
    best_name2 = 'expon'
    best_distribution2 = getattr(scipy.stats, best_name2)
    best_sse2 = np.inf
    best_params2 = (0.0, 1.0)
    for dist_name in dist_names:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Fit dist data
                dist = getattr(scipy.stats, dist_name)
                params = dist.fit(y2)
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(x2[:-1], loc=loc, scale=scale, *arg) * width
                sse = np.sum(np.power(result2[0] / cnt2 - pdf, 2.0))
                # identify if this distribution is better
                if best_sse2 > sse > 0:
                    best_distribution2 = dist
                    best_params2 = params
                    best_sse2 = sse
                    best_name2 = dist_name
        except Exception:
            pass

    # plot the best fit curve for Legit class
    if drawing == True:
        if x2[-1] > 40:
            x2_list = np.arange(x2[-1])
        elif x2[-1] > 5:
            x2_list = np.arange(0, x2[-1], 0.2)
        elif x2[-1] > 0.5:
            x2_list = np.arange(0, x2[-1], 0.05)
        else:
            x2_list = np.arange(0, x2[-1], 0.005)
        pdf_fitted2 = best_distribution2.pdf(x2_list, *best_params2[:-2], loc=best_params2[-2],
                                             scale=best_params2[-1]) * width

        # Draw histogram with density for Legit class
        plt.bar(x2[:-1], result2[0] / cnt2, width=width * 0.75, color='b', alpha=0.1, label='Legit Users')
        plt.plot(x2_list, pdf_fitted2, label='Legit: ' + best_name2, color='b', linestyle='--')
        if feature == 'between':
            plt.xlabel('Betweenness: FDS-' + method, fontsize=20)
#             plt.xscale('log')
        else:
            plt.xlabel('Social capital dimension: ' + feature, fontsize=20)
        plt.ylabel('Probability', fontsize=20)
        plt.tick_params(axis='both', labelsize=16)
        if dname == '1ks10kn' and feature == 'human':
            plt.legend(fontsize=18, loc='upper left', framealpha=0.3)
        else:
            plt.legend(fontsize=18)

        if dname == 'FakeLikers':
            plt.gcf().subplots_adjust(left=0.11)
            plt.gcf().subplots_adjust(top=1.0)
            plt.gcf().subplots_adjust(right=1.0)
            plt.savefig('resultliker/' + out + '/' + feature + '_' + method + '.png', dpi=300)
        if dname == 'Cresci15':
            plt.gcf().subplots_adjust(left=0.13)
            plt.gcf().subplots_adjust(top=0.98)
            plt.gcf().subplots_adjust(right=1.0)
            plt.savefig('resultcres/' + out + '/' + feature + '_' + method + '.png', dpi=300)
        if dname == 'INT':
            plt.gcf().subplots_adjust(left=0.11)
            plt.gcf().subplots_adjust(top=1.0)
            plt.gcf().subplots_adjust(right=1.0)
            plt.savefig('resultint/' + out + '/' + feature + '_' + method + '.png', dpi=300)
        elif dname == '1ks10kn':
            plt.gcf().subplots_adjust(left=0.13)
            plt.gcf().subplots_adjust(top=0.98)
            plt.gcf().subplots_adjust(right=1.0)
            plt.savefig('result1ks/' + out + '/' + feature + '_' + method + '.png', dpi=300)
        #else:
        plt.clf()
    return best_name1, best_distribution1, best_params1, best_name2, best_distribution2, best_params2

'''Return list of nodes need friends'''
def friend_pending(G):
    pending = []
    for (i, x) in G.nodes(data='full'):
        if x == False:
            pending.append(i)
    return pending

'''Return Graph capital data to likers for plotting'''
def capital_return(capital, G, feature='structural'):
    for i in G.nodes():
        val = G.nodes[i][feature]
        capital.at[i, feature] = val

'''Show how many nodes reached friends limit'''
def friend_full(G):
    f = 0
    for (i, x) in G.nodes(data='full'):
        if x == True:
            f = f + 1
    return f / G.order()

# find best fit probability density functions
def fitting_post(data, dname, feature='friends', method='SC', binnum=20, out='20'):
    xmax = max(data[feature])
    xmin = min(data[feature])

    # Fake Class
    result1 = np.histogram(data.loc[data['label'] == "Fake", :][feature], bins=binnum, range=(xmin, xmax))
    cnt1 = sum(data['label'] == "Fake")
    x1 = result1[1]
    y1 = data.loc[data['label'] == "Fake", :][feature]

    # Legit class
    result2 = np.histogram(data.loc[data['label'] == "Legit", :][feature], bins=binnum, range=(xmin, xmax))
    cnt2 = sum(data['label'] == "Legit")
    x2 = result2[1]
    y2 = data.loc[data['label'] == "Legit", :][feature]

    # draw histogram with same height
    height = max(np.max(result1[0]) / cnt1, np.max(result2[0]) / cnt2)
    tmp = round(height * 10) / 10
    if tmp < height:
        height = tmp + 0.1
    else:
        height = tmp

    dist_names = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'binom', 'bradford', 'burr', 'cauchy', 'chi',
                  'chi2', 'cosine',
                  'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                  'fisk',
                  'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto', 'gennorm', 'genexpon',
                  'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
                  'gumbel_l', 'halfcauchy', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss',
                  'halflogistic',
                  'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'levy_l',
                  # 'levy','levy_stable',
                  'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2',
                  'ncf',
                  'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal',
                  'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm',
                  'tukeylambda',
                  'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']  # 86

    width = result1[1][1] - result1[1][0]
    # plt.figure(figsize=(7,5))###
    plt.bar(x1[:-1], result1[0] / cnt1, width=width * 0.75, color='r', alpha=0.1, label='Fake')

    # -----------without pmf
    plt.bar(x2[:-1], result2[0] / cnt2, width=width * 0.75, color='b', alpha=0.1, label='Legit')
    #plt.xlabel('Social capital dimension: ' + feature, fontsize=18)
    plt.xlabel(dname + ' Degrees: ', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.legend(loc='upper right')
    #plt.title("Histogram of " + feature + " from method: " + method + " friend")
    if dname == 'FakeLikers':
        plt.savefig('resultliker/' + out + '/' + feature + '_' + method + '.png')
    elif dname == 'Cresci15':
        plt.savefig('resultcres/' + out + '/' + feature + '_' + method + '.png')
    elif dname == 'INT':
        plt.savefig('resultint/' + out + '/' + feature + '_' + method + '.png')
    elif dname == '1ks10kn':
        plt.savefig('result1ks/' + out + '/' + feature + '_' + method + '.png')
    plt.clf()
    # -----------

'''Adding likers data to Graph'''
def node_attributes(likers, G, feature):
    for (i, x) in enumerate(likers[feature].values):
        if feature == 'friends':
            G.nodes[likers.index[i]][feature] = int(x)
        else:
            G.nodes[likers.index[i]][feature] = x

'''Adding likers data to Graph'''
def node_attributes_edges(likers, G, dname):
    for (i, x) in enumerate(likers['friends'].values):
        G.nodes[likers.index[i]]['edge_interaction'] = math.ceil(x / 80)

'''Adding social capital data to Graph'''
def node_attributes_sc(likers, G):
    for (i, x) in enumerate(likers['human'].values):
        G.nodes[likers.index[i]]['capital'] = G.nodes[likers.index[i]]['human'] \
        + G.nodes[likers.index[i]]['cognitive'] + G.nodes[likers.index[i]]['relational']

'''Initialize graph nodes with attributes'''
def initialization_graph(likers, capital, n, dname):
    G = nx.empty_graph(n)
    # Fill features from likers
    nx.set_node_attributes(G, 0, 'edges')
    nx.set_node_attributes(G, False, 'full')
    node_attributes(likers, G, 'friends')
    node_attributes(likers, G, 'label')
    node_attributes_edges(likers, G, dname)

    # Fill in static social capital
    node_attributes(capital, G, 'human')
    node_attributes(capital, G, 'cognitive')
    node_attributes(capital, G, 'relational')
    node_attributes_sc(capital, G) #capital

    # Dynamic SC
    nx.set_node_attributes(G, 0, 'STC')
    nx.set_node_attributes(G, 0, 'CC')
    nx.set_node_attributes(G, 0, 'RC')
    nx.set_node_attributes(G, 0, 'SC')

    # Behavioral seeds
    node_attributes(capital, G, 'feeding')
    node_attributes(capital, G, 'posting')
    node_attributes(capital, G, 'feedback')
    node_attributes(capital, G, 'inviting')
    # credibility judgement
    nx.set_node_attributes(G, 1, 'experience') #normal user
    node_attributes(capital, G, 'competence') #normal user
    node_attributes(capital, G, 'deception') #attacker+compromised user
    # trust
    nx.set_node_attributes(G, 0, 'trust')
    nx.set_node_attributes(G, 1, 'fmax')
    nx.set_node_attributes(G, 1, 'bmax')

    # Attack model
    nx.set_node_attributes(G, '', 'phish')
    nx.set_node_attributes(G, 0, 'compromise')
    print(G.nodes[0])
    return G

'''Adding an new edge'''
def new_friend(G, u, v):
    G.add_edge(u, v, f=1, b=1) #initial feeding and feedback count
    # update friend count and full
    for i in [u, v]:
        G.nodes[i]['edges'] = G.nodes[i]['edges'] + 1
        if G.nodes[i]['edges'] >= G.nodes[i]['friends']:
            G.nodes[i]['full'] = True

'''Update social capital and structural capital'''
def social_capital_update(G, i):
    stc, cc, rc, tu = 0, 0, 0, 0
    neighbor_len = len(list(G.neighbors(i)))
    for j in list(G.neighbors(i)): #adding trust as weight
        trust = G.edges[i, j]['f'] / G.nodes[i]['fmax'] + G.edges[i, j]['b'] / G.nodes[i]['bmax']
        stc = stc + G.nodes[j]['human'] * trust/2
        cc = cc + G.nodes[j]['cognitive'] * trust/2
        rc = rc + G.nodes[j]['relational'] * trust/2
    G.nodes[i]['STC'] = stc / neighbor_len
    G.nodes[i]['CC'] = cc / neighbor_len
    G.nodes[i]['RC'] = rc / neighbor_len
    G.nodes[i]['SC'] = (G.nodes[i]['CC'] + G.nodes[i]['RC'] + G.nodes[i]['STC'])/ 3

'''Update social capital and structural capital'''
def social_capital_trust_update(G, i):
    stc, cc, rc, tu = 0, 0, 0, 0
    neighbor_len = len(list(G.neighbors(i)))
    for j in list(G.neighbors(i)): #adding trust as weight
        trust = G.edges[i, j]['f'] / G.nodes[i]['fmax'] + G.edges[i, j]['b'] / G.nodes[i]['bmax']
        stc = stc + G.nodes[j]['human'] * trust/2
        cc = cc + G.nodes[j]['cognitive'] * trust/2
        rc = rc + G.nodes[j]['relational'] * trust/2
        trustj = G.edges[i, j]['f'] / G.nodes[j]['fmax'] + G.edges[i, j]['b'] / G.nodes[j]['bmax']
        tu = tu + trustj / 2
    G.nodes[i]['STC'] = stc / neighbor_len
    G.nodes[i]['CC'] = cc / neighbor_len
    G.nodes[i]['RC'] = rc / neighbor_len
    G.nodes[i]['SC'] = (G.nodes[i]['CC'] + G.nodes[i]['RC'] + G.nodes[i]['STC'])/ 3
    G.nodes[i]['trust'] = tu / neighbor_len

'''Each user initial 5 friends'''
def initial_friend_SC(G, feature, run, dname):
    rnd = int(1000*random.random())
    print("initial SC seed:", rnd)
    random.seed(rnd)

    pending = friend_pending(G)
    random.shuffle(pending)
    full_nodes = set(G.nodes()) - set(pending)
    for i in range(5):
        for j in pending:
            if G.nodes[j]['edges'] / G.nodes[j]['friends'] > 0.4 or G.nodes[j]['edges'] > 6: #leave space for very small number of friends
                continue

            # find all possible k
            friend_j = list(G.neighbors(j))
            list_k = pending[:] #deepcopy
            list_k.remove(j)
            list_k = list(set(list_k) - set(friend_j))
            if len(list_k) == 0:
                G.nodes[j]['full'] = True
                continue
            # collect all k to node_list
            node_list0 = {}
            for k in list_k:
                if G.nodes[k]['full'] == False and G.nodes[k]['edges'] < 7:
                    node_list0[k] = G.nodes[k][feature]
            if len(node_list0) == 0:
                G.nodes[j]['full'] = True
                continue
            node_list = {}
            for k in node_list0.keys():
                node_list[k] = node_list0[k]
            if len(node_list) == 0:
                continue
            # rank node_list and pick top 1
            node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
            flag = 0
            ranked = list(node_list.keys())  # ascending
            #select top 1 value
            while flag == 0:
                if len(node_list) == 0 or G.nodes[j]['full'] == True:
                    break
                ranked = list(node_list.keys())  # ascending
                f = ranked[-1]
                if G.nodes[f]['full'] == False and G.nodes[f]['edges'] < 7:
                    new_friend(G, j, f)
                    flag = 1
                del node_list[f]

        # simulate positive interaction
        for j in list(G.nodes()):
            if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                for fb in list(G.neighbors(j)): #update feeding
                    G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                    if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                        G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                    if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                        G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                    if G.nodes[fb]['feedback'] >= random.random(): #updating feedback
                        G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                        if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                            G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                        if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                            G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
        pending = friend_pending(G)
        random.shuffle(pending)
    #after 5 times
    for i in list(G.nodes()): #update sc and trust values
        social_capital_trust_update(G, i)
    print(G.size())
    return G

'''Each user initial 5 friends'''
def initial_friend_TR(G, run, dname):
    rnd = int(1000*random.random())
    print("initial TR seed:", rnd)
    random.seed(rnd)

    pending = friend_pending(G)
    random.shuffle(pending)
    full_nodes = set(G.nodes()) - set(pending)
    for i in range(5):
        for j in pending:
            if G.nodes[j]['edges'] / G.nodes[j]['friends'] > 0.4 or G.nodes[j]['edges'] > 6: #leave space for very small number of friends
                continue

            # find all possible k
            friend_j = list(G.neighbors(j))
            list_k = pending[:] #deepcopy
            list_k.remove(j)
            list_k = list(set(list_k) - set(friend_j))
            if len(list_k) == 0:
                G.nodes[j]['full'] = True
                continue
            # collect all k to node_list
            node_list0 = {}
            for k in list_k:
                if G.nodes[k]['full'] == False:
                    node_list0[k] = G.nodes[k]['feeding'] + G.nodes[k]['feedback']
            if len(node_list0) == 0:
                G.nodes[j]['full'] = True
                continue
            node_list = {}
            for k in node_list0.keys():
                node_list[k] = node_list0[k]
            if len(node_list) == 0:
                continue
            # rank node_list and pick top 1
            node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
            flag = 0
            ranked = list(node_list.keys())  # ascending
            #select top 1 value
            while flag == 0:
                if len(node_list) == 0 or G.nodes[j]['full'] == True:
                    break
                ranked = list(node_list.keys())  # ascending
                f = ranked[-1]
                if G.nodes[f]['full'] == False:
                    new_friend(G, j, f)
                    flag = 1
                del node_list[f]

        # simulate positive interaction
        for j in list(G.nodes()):
            if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                for fb in list(G.neighbors(j)): #update feeding
                    G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                    if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                        G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                    if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                        G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                    if G.nodes[fb]['feedback'] >= random.random(): #updating feedback
                        G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                        if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                            G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                        if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                            G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
        pending = friend_pending(G)
        random.shuffle(pending)
    #after 5 times
    for i in list(G.nodes()): #update sc and trust values
        social_capital_trust_update(G, i)
    print(G.size())
    return G

'''Each user initial 5 friends'''
def initial_friend_SA(G, run, dname):
    rnd = int(1000*random.random())
    print("initial SA seed:", rnd)
    random.seed(rnd)

    # adjacency matrix to save similarity
    size = max(G.nodes()) + 1
    product_matrix = np.zeros((size, size))
    for x in G.nodes():
        dot_x = [G.nodes[x]['human'], G.nodes[x]['cognitive'], G.nodes[x]['relational']]
        for y in G.nodes():
            if y > x:
                dot_y = [G.nodes[y]['human'], G.nodes[y]['cognitive'], G.nodes[y]['relational']]
                mut = np.dot(dot_x, dot_y) / np.linalg.norm(dot_x) / np.linalg.norm(dot_y)
                product_matrix[x][y] = mut

    pending = friend_pending(G)
    random.shuffle(pending)
    full_nodes = set(G.nodes()) - set(pending)
    for i in range(5):
        for j in pending:
            if G.nodes[j]['edges'] / G.nodes[j]['friends'] > 0.4 or G.nodes[j]['edges'] > 6: #leave space for very small number of friends
                continue

            # find all possible k
            friend_j = list(G.neighbors(j))
            list_k = pending[:] #deepcopy
            list_k.remove(j)
            list_k = list(set(list_k) - set(friend_j))
            if len(list_k) == 0:
                G.nodes[j]['full'] = True
                continue
            # collect all k to node_list
            node_list0 = {}
            for k in list_k:
                if G.nodes[k]['full'] == False:
                    node_list0[k] = product_matrix[min(j, k)][max(j, k)]
            if len(node_list0) == 0:
                G.nodes[j]['full'] = True
                continue
            node_list = {}
            for k in node_list0.keys():
                    node_list[k] = node_list0[k]
            if len(node_list) == 0:
                continue
            # rank node_list and pick top 1
            node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
            flag = 0
            ranked = list(node_list.keys())  # ascending
            #select top 1 value
            while flag == 0:
                if len(node_list) == 0 or G.nodes[j]['full'] == True:
                    break
                ranked = list(node_list.keys())  # ascending
                f = ranked[-1]
                if G.nodes[f]['full'] == False and G.nodes[j]['edges'] < 7:
                    new_friend(G, j, f)
                    flag = 1
                del node_list[f]

        # simulate positive interaction
        for j in list(G.nodes()):
            if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                for fb in list(G.neighbors(j)): #update feeding
                    G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                    if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                        G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                    if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                        G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                    if G.nodes[fb]['feedback'] >= random.random(): #updating feedback
                        G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                        if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                            G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                        if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                            G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
        pending = friend_pending(G)
        random.shuffle(pending)
    #after 5 times
    for i in list(G.nodes()): #update sc and trust values
        social_capital_trust_update(G, i)
    print(G.size())
    return G

'''Each user initial 5 friends'''
def initial_friend_TP(G, topics, run, dname):
    rnd = int(1000*random.random())
    print("initial TP seed:", rnd)
    random.seed(rnd)

    # adjacency matrix to save similarity
    size = max(G.nodes()) + 1
    product_matrix = np.zeros((size, size))
    for x in G.nodes():
        dot_x = topics.loc[x]
        for y in G.nodes():
            if y > x:
                dot_y = topics.loc[y]
                mut = np.dot(dot_x, dot_y) / np.linalg.norm(dot_x) / np.linalg.norm(dot_y)
                product_matrix[x][y] = mut

    pending = friend_pending(G)
    random.shuffle(pending)
    full_nodes = set(G.nodes()) - set(pending)
    for i in range(5):
        for j in pending:
            if G.nodes[j]['edges'] / G.nodes[j]['friends'] > 0.4 or G.nodes[j]['edges'] > 6: #leave space for very small number of friends
                continue

            # find all possible k
            friend_j = list(G.neighbors(j))
            list_k = pending[:] #deepcopy
            list_k.remove(j)
            list_k = list(set(list_k) - set(friend_j))
            if len(list_k) == 0:
                G.nodes[j]['full'] = True
                continue
            # collect all k to node_list
            node_list0 = {}
            for k in list_k:
                if G.nodes[k]['full'] == False and G.nodes[j]['edges'] < 7:
                    node_list0[k] = product_matrix[min(j, k)][max(j, k)]
            if len(node_list0) == 0:
                G.nodes[j]['full'] = True
                continue
            node_list = {}
            for k in node_list0.keys():
                    node_list[k] = node_list0[k]
            if len(node_list) == 0:
                continue
            # rank node_list and pick top 1
            node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
            flag = 0
            ranked = list(node_list.keys())  # ascending
            #select top 1 value
            while flag == 0:
                if len(node_list) == 0 or G.nodes[j]['full'] == True:
                    break
                ranked = list(node_list.keys())  # ascending
                f = ranked[-1]
                if G.nodes[f]['full'] == False:
                    new_friend(G, j, f)
                    flag = 1
                del node_list[f]

        # simulate positive interaction
        for j in list(G.nodes()):
            if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                for fb in list(G.neighbors(j)): #update feeding
                    G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                    if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                        G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                    if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                        G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                    if G.nodes[fb]['feedback'] >= random.random(): #updating feedback
                        G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                        if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                            G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                        if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                            G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
        pending = friend_pending(G)
        random.shuffle(pending)
    #after 5 times
    for i in list(G.nodes()): #update sc and trust values
        social_capital_trust_update(G, i)
    print(G.size())
    return G

'''Update capital data from G and plot probability distribution'''
def post_process(G, capital, i, method, dname, out):
    # print stats
    print(method, 'Edges:', G.size())
    print(method, 'Non-full nodes:',
          len([n for n, v in G.nodes(data=True) if v['full'] == False]))  # print none full nodes
    print(method, 'Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())

    # Add social capital values to capital
    capital_return(capital, G, 'edges')
    capital_return(capital, G, 'STC')
    capital_return(capital, G, 'CC')
    capital_return(capital, G, 'RC')
    capital_return(capital, G, 'SC')
    print('# SC=0: ', sum(capital['SC'] == 0.0))

    # # Draw probability distribution
    # if i == 0 and method == 'SC':
    #     fitting_post(capital, dname, 'cognitive', method, 50, out)
    #     fitting_post(capital, dname, 'relational', method, 50, out)
    #     fitting_post(capital, dname, 'human', method, 50, out)
    #     fitting_post(capital, dname, 'RH', method, 50, out)
    #     fitting_post(capital, dname, 'RHC', method, 50, out)
    #     fitting_post(capital, dname, 'edges', method, 50, out)
    #     fitting_post(capital, dname, 'SC', method, 50, out)
    #     fitting_post(capital, dname, 'STC', method, 50, out)
    #     fitting_post(capital, dname, 'CC', method, 50, out)
    #     fitting_post(capital, dname, 'RC', method, 50, out)
    # elif i == 0:
    #     fitting_post(capital, dname, 'edges', method, 50, out)
    #     fitting_post(capital, dname, 'SC', method, 50, out)
    #     fitting_post(capital, dname, 'STC', method, 50, out)
    #     fitting_post(capital, dname, 'CC', method, 50, out)
    #     fitting_post(capital, dname, 'RC', method, 50, out)

    #print SC values
    for feature in ['STC', 'CC', 'RC', 'SC']:
        if dname == 'Cresci15':
            with open('resultcres/' + out + '/' + feature + '_' + method + '.txt', 'a') as f:
                f.write(str(G.nodes(data=feature)) + '\n')
        elif dname == '1ks10kn':
            with open('result1ks/' + out + '/' + feature + '_' + method + '.txt', 'a') as f:
                f.write(str(G.nodes(data=feature)) + '\n')
    return capital

'''Defense for four social deception attacks'''
def defense(G, atta, i, records, j):  # j:attacker, i:target, f:friends
    start_time = time.time()
    #random.seed(j)
    random.seed(int(start_time))
    G.nodes[i][atta] = 'E'
    flag = True
    # defense: detect and post to friends
    cred = math.exp(-G.nodes[j]['deception']/(G.nodes[i]['competence']*G.nodes[i]['experience']))
    if cred >= random.random():
        G.nodes[i][atta] = 'R'
        G.nodes[i]['experience'] = G.nodes[i]['experience'] + 1
        records['sir'][i] = records['sir'][i] + 1
        flag = False
        # post to friends
        if G.nodes[i]['posting'] >= random.random():  # small chance to post
            for f in list(G.neighbors(i)):
                # friends increase the experience of attackers
                if f and not isinstance(G.nodes[j][atta], int):
                    G.nodes[f]['experience'] = G.nodes[f]['experience'] + 1
    else:  # defense: fail to detect and ask for friend
        if G.nodes[i]['posting'] >= random.random():  # small chance to post
            for f in list(G.neighbors(i)):
                # friend comment back to help detect
                if f and (G.nodes[f][atta] == 'S' or G.nodes[f][atta] == 'R') and G.nodes[f]['feedback'] >= random.random():
                    cred_f = math.exp(-G.nodes[j]['deception'] / (G.nodes[f]['competence'] * G.nodes[f]['experience']))
                    if cred_f >= random.random():
                        records['ir'][i] = records['ir'][i] + 1
                        G.nodes[i][atta] = 'R'
                        G.nodes[i]['experience'] = G.nodes[i]['experience'] + 1
                        G.nodes[f]['experience'] = G.nodes[f]['experience'] + 1
                        flag = False
                        break
    return flag

'''Attack find the targets'''
def interaction_ATTACK(G, likers, dname, atta, j, pas, records, i):
    start_time = time.time()
    rnd = int(i * 10)
    seed(rnd)

    # one-time attack-defense-update
    neighbors = list(G.neighbors(j))
    f_list = [x for x, y in G.nodes(data=True) if (not isinstance(G.nodes[x][atta], int) and x in neighbors)]
    random.shuffle(f_list)
    victims = math.ceil(len(f_list) * pas)
    if len(f_list) == 0:
        return G
    elif atta == 'phish':  # phishing: select at random
        targets = choice(f_list, victims, replace=False)  # attack p_as percent friends

    if type(targets) is list or type(targets) is np.ndarray:
        count_i = 0
        count_a = 0
        for target in targets:
            if G.nodes[target][atta] == 'S' or G.nodes[target][atta] == 'E':
                count_a = count_a + 1
                # defense: detect and post to friends/ fail to detect and ask for friend
                flag = defense(G, atta, target, records, j)
                records['i'][target] = records['i'][target] + 1
                if flag == True: # Infected
                    count_i = count_i + 1
                    G.edges[j, target]['f'] = G.edges[j, target]['f'] + 1
                    G.edges[j, target]['b'] = G.edges[j, target]['b'] + 1
                else: #defense successful
                    G.nodes[j][atta] = G.nodes[j][atta] + 1 #report the attacker
                    if G.nodes[j]['label'] == True:
                        G.remove_edge(j, target) #terminate friendship with compromised account
                    else:
                        G.edges[j, target]['f'] = G.edges[j, target]['f'] + 1
            else: # R user can ignore the attack and report it
                G.nodes[j][atta] = G.nodes[j][atta] + 1
                G.nodes[target]['experience'] = G.nodes[target]['experience'] + 1
    else:  # scalar
        print("ERROR")
    return G

'''Update SIR for four attacks after one interaction time'''
def sir_process_update(G, attack_sir, run, feat, i):
    adding = 100
    snum = len([x for x, y in G.nodes(data=True) if y['phish'] == 'S'])
    enum = len([x for x, y in G.nodes(data=True) if y['phish'] == 'E'])
    inum = len([x for x, y in G.nodes(data=True) if (y['phish'] == 0 and y['label'] == 'Legit')])
    rnum = len([x for x, y in G.nodes(data=True) if y['phish'] == 'R'])
    sir = snum + inum + rnum + enum
    attack_sir[(i-adding)*4] = snum / sir
    attack_sir[(i-adding)*4+1] = inum / sir
    attack_sir[(i-adding)*4+2] = rnum / sir
    attack_sir[(i - adding) * 4 + 3] = enum / sir

'''Update SIR for four attacks after one interaction time'''
def sir_state_change(records, attack_state, i):
    adding = 100
    attack_state[(i-adding) * 4] = len([(x,y) for x,y in records['i'].items() if y>0])
    attack_state[(i-adding) * 4 + 1] = len([x for x,y in records['sr'].items() if y>0])
    attack_state[(i-adding) * 4 + 2] = len([x for x,y in records['ir'].items() if y>0])
    attack_state[(i-adding) * 4 + 3] = len([x for x,y in records['sir'].items() if y>0])

'''Simulate time-series behaviors of users-- Social capital dimensions'''
def interaction_SC(G, dname, likers, feature, p_as, run, attack_sir, attack_state, attack_report, feat):
    start_time = time.time()
    rnd = int(start_time)
    print("run seed:", rnd)
    random.seed(rnd)

    #recording attacks
    records = {}
    records['i'] = {no: 0 for no in G.nodes()}
    records['sr'] = {no: 0 for no in G.nodes()}
    records['ir'] = {no: 0 for no in G.nodes()}
    records['sir'] = {no: 0 for no in G.nodes()}
    invite = {no: 0 for no in G.nodes()}

    for _, y in G.nodes(data=True):
        attack = 'phish'
        if y['label'] == 'Legit':
            y[attack] = 'S'
        else:
            y[attack] = 0

    size_prev = G.size()
    for i in range(105):
        size_invite = 0
        if i < 100:  # inviting friends only
            pending = friend_pending(G)
            random.shuffle(pending)
            full_nodes = set(G.nodes()) - set(pending)
            ###
            edge_cur = [0 for x in pending] #can change to attribute of G
            for j in pending:
                idx_j = pending.index(j)
                limit_j = G.nodes[j]['edge_interaction']
                if G.nodes[j]['full'] == True:
                    continue
                ###
                a = random.random()
                if G.nodes[j]['inviting'] >= a:
                    size_invite = size_invite + 1
                    invite[j] = invite[j] + 1
                    # find all possible k
                    friend_j = list(G.neighbors(j))
                    list_k = deepcopy(pending)
                    list_k.remove(j)
                    list_k = list(set(list_k) - set(friend_j))
                    if len(list_k) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    # collect all k to node_list
                    node_list0 = {}
                    for k in list_k:
                        if G.nodes[k]['full'] == False:
                            node_list0[k] = G.nodes[k][feature]
                    if len(node_list0) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    node_list = {}
                    for k in node_list0.keys():
                        if edge_cur[pending.index(k)] < G.nodes[k]['edge_interaction']:
                            node_list[k] = node_list0[k]
                    if len(node_list) == 0:
                        continue

                    # rank node_list and pick top 1 or 10/20
                    node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
                    flag = 0
                    ranked = list(node_list.keys())  # ascending
                    length = len(ranked)

                    # #select top 1 value
                    while edge_cur[idx_j] < limit_j or (edge_cur[idx_j] == limit_j and flag == 0):
                        if len(node_list) == 0 or G.nodes[j]['full'] == True:
                            break
                        ranked = list(node_list.keys())  # ascending
                        f = ranked[-1]
                        if G.nodes[f]['label'] == 'Legit' and G.nodes[j][feature] < 1.0 * node_list[f]:
                            del node_list[f]
                            continue
                        if edge_cur[pending.index(f)] < G.nodes[f]['edge_interaction'] and G.nodes[f]['full'] == False:
                            new_friend(G, j, f)
                            flag = 1
                            edge_cur[idx_j] = edge_cur[idx_j] + 1
                            edge_cur[pending.index(f)] = edge_cur[pending.index(f)] + 1
                        del node_list[f]

            size_prev = G.size()
            full_nodes_new = set(G.nodes()) - set(friend_pending(G))
            full_nodes_iter = full_nodes_new - full_nodes
            # print(G.size())

            if i < 80:
                # simulate positive interaction
                for j in list(G.nodes()):
                    if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                        for fb in list(G.neighbors(j)):  # update feeding
                            G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                            if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                                G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                            if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                                G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                            if G.nodes[fb]['feedback'] >= random.random():  # updating feedback
                                G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                                if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                                    G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                                if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                                    G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
                # update sc values after each run
                for j in list(G.nodes()):
                    social_capital_update(G, j)
        else: # attack and defense
            nodes = list(G.nodes())
            random.shuffle(nodes)
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):
                    G = interaction_ATTACK(G, likers, dname, 'phish', j, p_as, records, i)  # phishing

            # stats collection
            count = 0 # reported attackers
            for j in nodes:
                if G.nodes[j]['label'] == 'Fake' and G.nodes[j]['phish'] >= 3:
                    count = count + 1
            attack_report[i-100] = count

            sir_state_change(records, attack_state, i)

            # post-attack updates: attacker report
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):  #attackers
                    # removed the reported attackers from the network/ compromised user back to S
                    if G.nodes[j]['phish'] >= 3:
                        if G.nodes[j]['label'] == True:  # compromised account
                            G.nodes[j]['phish'] = 'R'  # change password
                            G.nodes[j]['experience'] = G.nodes[j]['experience'] + 1
                        else:  # remove the attacker
                            nb = list(G.neighbors(j))
                            for k in nb:
                                G.remove_edge(j, k)
                else: #normal users
                    if G.nodes[j]['phish'] == 'E' and random.random() <= 0.2: #E to attacker I, compromised account
                        G.nodes[j]['phish'] = 0 #20% chance to be compromised accounts
            sir_process_update(G, attack_sir, run, feat, i) #update S E I R


        # printing
        if i == 100:
            invite_times = {}
            invite_friends = {}
            for x in sorted(invite.keys()):
                if invite[x] > 0:
                    invite_times[x] = invite[x]
                    invite_friends[x] = G.nodes[x]['friends']
            print(len(invite_times))

        if (i + 1) % 20 == 0:
            print("%s edges at %d :%d" % (feature, i, G.size()))
    print((time.time() - start_time) / 60)

    sum = 0
    diff = 0
    fake_edges = {}
    fake_friends = {}
    for x in G.nodes():
        if G.nodes[x]['label'] == 'Fake':
            diff = diff + abs(G.nodes[x]['friends'] - G.nodes[x]['edges'])
            sum = sum + G.nodes[x]['friends']
            fake_edges[x] = G.nodes[x]['edges']
            fake_friends[x] = G.nodes[x]['friends']
    print("sum of fake friends:", sum)
    print("sumdiff of fake friends:", diff)
    return G

'''Simulate time-series behaviors of users-- Topics similarity'''
def interaction_TP(G, dname, likers, topics, p_as, run, attack_sir, attack_state, attack_report, feat):
    start_time = time.time()
    rnd = int(start_time)
    print("run seed:", rnd)
    random.seed(rnd)

    #recording attacks
    records = {}
    records['i'] = {no: 0 for no in G.nodes()}
    records['sr'] = {no: 0 for no in G.nodes()}
    records['ir'] = {no: 0 for no in G.nodes()}
    records['sir'] = {no: 0 for no in G.nodes()}
    invite = {no: 0 for no in G.nodes()}

    #random.seed(int(start_time))
    for _, y in G.nodes(data=True):
        for attack in ['phish', 'falseinfo', 'profile', 'humanatt']:
            if y['label'] == 'Legit':
                y[attack] = 'S'
            else:
                y[attack] = 0

    # adjacency matrix to save similarity
    size = max(G.nodes()) + 1
    product_matrix = np.zeros((size, size))
    for x in G.nodes():
        dot_x = topics.loc[x]
        for y in G.nodes():
            if y > x:
                dot_y = topics.loc[y]
                mut = np.dot(dot_x, dot_y) / np.linalg.norm(dot_x) / np.linalg.norm(dot_y)
                product_matrix[x][y] = mut

    size_prev = G.size()
    for i in range(105):
        size_invite = 0
        if i < 100:  # inviting friends only
            pending = friend_pending(G)
            random.shuffle(pending)

            full_nodes = set(G.nodes()) - set(pending)
            edge_cur = [0 for x in pending] #can change to attribute of G
            for j in pending:
                idx_j = pending.index(j)
                limit_j = G.nodes[j]['edge_interaction']
                if G.nodes[j]['full'] == True:
                    continue
                ###
                a = random.random()
                if G.nodes[j]['inviting'] >= a:
                    size_invite = size_invite + 1
                    invite[j] = invite[j] + 1
                    # find all possible k
                    friend_j = list(G.neighbors(j))
                    list_k = deepcopy(pending)
                    list_k.remove(j)
                    list_k = list(set(list_k) - set(friend_j))
                    if len(list_k) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    node_list0 = {}
                    for k in list_k:
                        if G.nodes[k]['full'] == False:
                            node_list0[k] = product_matrix[min(j, k)][max(j, k)]
                    if len(node_list0) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    node_list = {}
                    for k in node_list0.keys():
                        if edge_cur[pending.index(k)] < G.nodes[k]['edge_interaction']:
                            node_list[k] = node_list0[k]
                    if len(node_list) == 0:
                        continue

                    # rank node_list and pick top 1
                    node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
                    flag = 0
                    while edge_cur[idx_j] < limit_j or (edge_cur[idx_j] == limit_j and flag == 0):
                        if len(node_list) == 0 or G.nodes[j]['full'] == True:
                            break
                        ranked = list(node_list.keys())  # ascending
                        f = ranked[-1]
                        if G.nodes[f]['label'] == 'Legit':
                            tp_max = 0
                            for k in G.neighbors(f):
                                if tp_max < product_matrix[min(f, k)][max(f, k)]:
                                    tp_max = product_matrix[min(f, k)][max(f, k)]
                            if node_list[f] < 0.2 * tp_max:
                                del node_list[f]
                                continue
                        ###
                        if edge_cur[pending.index(f)] < G.nodes[f]['edge_interaction'] and G.nodes[f]['full'] == False:
                            new_friend(G, j, f)
                            flag = 1
                            edge_cur[idx_j] = edge_cur[idx_j] + 1
                            edge_cur[pending.index(f)] = edge_cur[pending.index(f)] + 1
                        del node_list[f]

            size_prev = G.size()
            full_nodes_new = set(G.nodes()) - set(friend_pending(G))
            full_nodes_iter = full_nodes_new - full_nodes

            if i < 80:
                # simulate positive interaction
                for j in list(G.nodes()):
                    if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                        for fb in list(G.neighbors(j)):  # update feeding
                            G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                            if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                                G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                            if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                                G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                            if G.nodes[fb]['feedback'] >= random.random():  # updating feedback
                                G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                                if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                                    G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                                if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                                    G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
                # update sc values after each run
                for j in list(G.nodes()):
                    social_capital_update(G, j)
        else: # attack and defense
            nodes = list(G.nodes())
            random.shuffle(nodes)
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):
                    G = interaction_ATTACK(G, likers, dname, 'phish', j, p_as, records, i)  # phishing

            # stats collection
            count = 0 # reported attackers
            for j in nodes:
                if G.nodes[j]['label'] == 'Fake' and G.nodes[j]['phish'] >= 3:  #deception == 2.5
                    count = count + 1
            attack_report[i-100] = count
            sir_state_change(records, attack_state, i)

            # post-attack updates: attacker report and user transformation
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):  #attackers
                    # removed the reported attackers from the network/ compromised user back to S
                    if G.nodes[j]['phish'] >= 3:
                        if G.nodes[j]['label'] == True:  # compromised account
                            G.nodes[j]['phish'] = 'R'  # change password
                            G.nodes[j]['experience'] = G.nodes[j]['experience'] + 1
                        else:  # remove the attacker
                            nb = list(G.neighbors(j))
                            for k in nb:
                                G.remove_edge(j, k)
                else: #normal users
                    if G.nodes[j]['phish'] == 'E' and random.random() <= 0.2: #I to attacker, compromised account
                        G.nodes[j]['phish'] = 0
            sir_process_update(G, attack_sir, run, feat, i)
        # printing
        if i == 100:
            invite_times = {}
            invite_friends = {}
            for x in sorted(invite.keys()):
                if invite[x] > 0:
                    invite_times[x] = invite[x]
                    invite_friends[x] = G.nodes[x]['friends']
            print(len(invite_times))

        if (i + 1) % 20 == 0:
            print("TP edges at %d :%d" % (i, G.size()))
    print((time.time() - start_time) / 60)

    sum = 0
    diff = 0
    fake_edges = {}
    fake_friends = {}
    for x in G.nodes():
        if G.nodes[x]['label'] == 'Fake':
            diff = diff + abs(G.nodes[x]['friends'] - G.nodes[x]['edges'])
            sum = sum + G.nodes[x]['friends']
            fake_edges[x] = G.nodes[x]['edges']
            fake_friends[x] = G.nodes[x]['friends']
    print("sum of fake friends:", sum)
    print("sumdiff of fake friends:", diff)
    return G

'''Simulate time-series behaviors of users-- Social attributes'''
def interaction_SA(G, dname, likers, p_as, run, attack_sir, attack_state, attack_report, feat):
    start_time = time.time()
    rnd = int(start_time)
    print("run seed:", rnd)
    random.seed(rnd)

    #recording attacks
    records = {}
    records['i'] = {no: 0 for no in G.nodes()}
    records['sr'] = {no: 0 for no in G.nodes()}
    records['ir'] = {no: 0 for no in G.nodes()}
    records['sir'] = {no: 0 for no in G.nodes()}
    invite = {no: 0 for no in G.nodes()}

    #random.seed(int(start_time))
    for _, y in G.nodes(data=True):
        for attack in ['phish', 'falseinfo', 'profile', 'humanatt']:
            if y['label'] == 'Legit':
                y[attack] = 'S'
            else:
                y[attack] = 0

    # adjacency matrix to save similarity
    size = max(G.nodes()) + 1
    product_matrix = np.zeros((size, size))
    for x in G.nodes():
        dot_x = [G.nodes[x]['human'], G.nodes[x]['cognitive'], G.nodes[x]['relational']]
        for y in G.nodes():
            if y > x:
                dot_y = [G.nodes[y]['human'], G.nodes[y]['cognitive'], G.nodes[y]['relational']]
                mut = np.dot(dot_x, dot_y) / np.linalg.norm(dot_x) / np.linalg.norm(dot_y)
                product_matrix[x][y] = mut

    size_prev = G.size()
    for i in range(105):
        size_invite = 0
        if i < 100:  # inviting friends only
            pending = friend_pending(G)
            random.shuffle(pending)

            full_nodes = set(G.nodes()) - set(pending)
            edge_cur = [0 for x in pending] #can change to attribute of G
            for j in pending:
                idx_j = pending.index(j)
                limit_j = G.nodes[j]['edge_interaction']
                if G.nodes[j]['full'] == True:
                    continue

                a = random.random()
                if G.nodes[j]['inviting'] >= a:
                    size_invite = size_invite + 1
                    invite[j] = invite[j] + 1
                    # find all possible k
                    friend_j = list(G.neighbors(j))
                    list_k = deepcopy(pending)
                    list_k.remove(j)
                    list_k = list(set(list_k) - set(friend_j))
                    if len(list_k) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    node_list0 = {}
                    for k in list_k:
                        if G.nodes[k]['full'] == False:
                            node_list0[k] = product_matrix[min(j, k)][max(j, k)]
                    if len(node_list0) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    node_list = {}
                    for k in node_list0.keys():
                        if edge_cur[pending.index(k)] < G.nodes[k]['edge_interaction']:
                            node_list[k] = node_list0[k]
                    if len(node_list) == 0:
                        continue

                    # rank node_list and pick top 1
                    node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
                    flag = 0
                    while edge_cur[idx_j] < limit_j or (edge_cur[idx_j] == limit_j and flag == 0):
                        if len(node_list) == 0 or G.nodes[j]['full'] == True:
                            break
                        ranked = list(node_list.keys())  # ascending
                        f = ranked[-1]
                        if G.nodes[f]['label'] == 'Legit':
                            as_max = 0
                            for k in G.neighbors(f):
                                if as_max < product_matrix[min(f, k)][max(f, k)]:
                                    as_max = product_matrix[min(f, k)][max(f, k)]
                            if node_list[f] < 0.6 * as_max:
                                del node_list[f]
                                continue

                        if edge_cur[pending.index(f)] < G.nodes[f]['edge_interaction'] and G.nodes[f]['full'] == False:
                            new_friend(G, j, f)
                            flag = 1
                            edge_cur[idx_j] = edge_cur[idx_j] + 1
                            edge_cur[pending.index(f)] = edge_cur[pending.index(f)] + 1
                        del node_list[f]

            size_prev = G.size()
            full_nodes_new = set(G.nodes()) - set(friend_pending(G))
            full_nodes_iter = full_nodes_new - full_nodes

            if i<80:
                # simulate positive interaction
                for j in list(G.nodes()):
                    if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                        for fb in list(G.neighbors(j)):  # update feeding
                            G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                            if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                                G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                            if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                                G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                            if G.nodes[fb]['feedback'] >= random.random():  # updating feedback
                                G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                                if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                                    G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                                if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                                    G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
                # update sc values after each run
                for j in list(G.nodes()):
                    social_capital_update(G, j)
        else: # attack and defense
            nodes = list(G.nodes())
            random.shuffle(nodes)
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):
                    G = interaction_ATTACK(G, likers, dname, 'phish', j, p_as, records, i)  # phishing


            # stats collection
            count = 0 # reported attackers
            for j in nodes:
                if G.nodes[j]['label'] == 'Fake' and G.nodes[j]['phish'] >= 3:  #deception == 2.5
                    count = count + 1
            attack_report[i-100] = count
            sir_state_change(records, attack_state, i)

            # post-attack updates: attacker report and user transformation
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):  #attackers
                    # removed the reported attackers from the network/ compromised user back to S
                    if G.nodes[j]['phish'] >= 3:
                        if G.nodes[j]['label'] == True:  # compromised account
                            G.nodes[j]['phish'] = 'R'  # change password
                            G.nodes[j]['experience'] = G.nodes[j]['experience'] + 1
                        else:  # remove the attacker
                            nb = list(G.neighbors(j))
                            for k in nb:
                                G.remove_edge(j, k)
                else: #normal users
                    if G.nodes[j]['phish'] == 'E' and random.random() <= 0.2: #I to attacker, compromised account
                        G.nodes[j]['phish'] = 0
            sir_process_update(G, attack_sir, run, feat, i)
        # printing
        if i == 100:
            invite_times = {}
            invite_friends = {}
            for x in sorted(invite.keys()):
                if invite[x] > 0:
                    invite_times[x] = invite[x]
                    invite_friends[x] = G.nodes[x]['friends']
            print(len(invite_times))

        if (i + 1) % 20 == 0:
            print("SA edges at %d :%d" % (i, G.size()))
    print((time.time() - start_time) / 60)

    sum = 0
    diff = 0
    fake_edges = {}
    fake_friends = {}
    for x in G.nodes():
        if G.nodes[x]['label'] == 'Fake':
            diff = diff + abs(G.nodes[x]['friends'] - G.nodes[x]['edges'])
            sum = sum + G.nodes[x]['friends']
            fake_edges[x] = G.nodes[x]['edges']
            fake_friends[x] = G.nodes[x]['friends']
    print("sum of fake friends:", sum)
    print("sumdiff of fake friends:", diff)
    return G

'''Simulate time-series behaviors of users-- Trust friend'''
def interaction_TR(G, dname, likers, feature, p_as, run, attack_sir, attack_state, attack_report, feat):
    start_time = time.time()
    rnd = int(start_time)
    print("run seed:", rnd)
    random.seed(rnd)

    # recording attacks
    records = {}
    records['i'] = {no: 0 for no in G.nodes()}
    records['sr'] = {no: 0 for no in G.nodes()}
    records['ir'] = {no: 0 for no in G.nodes()}
    records['sir'] = {no: 0 for no in G.nodes()}
    invite = {no: 0 for no in G.nodes()}

    # set SIR(legit) or reported time(attacker)
    for _, y in G.nodes(data=True):
        for attack in ['phish']:
            if y['label'] == 'Legit':
                y[attack] = 'S' #for legit users
            else:
                y[attack] = 0 #for attackers

    # Dynamic interactions
    size_prev = G.size()
    for i in range(105):
        size_invite = 0
        if i < 100:  # no attacks
            pending = friend_pending(G)
            random.shuffle(pending)
            full_nodes = set(G.nodes()) - set(pending)
            edge_cur = [0 for x in pending] #can change to attribute of G
            for j in pending:
                idx_j = pending.index(j)
                limit_j = G.nodes[j]['edge_interaction']
                if G.nodes[j]['full'] == True:
                    continue
                a = random.random()
                if G.nodes[j]['inviting'] >= a:
                    size_invite = size_invite + 1
                    invite[j] = invite[j] + 1
                    # find all possible k
                    friend_j = list(G.neighbors(j))
                    list_k = deepcopy(pending)
                    list_k.remove(j)
                    list_k = list(set(list_k) - set(friend_j))
                    if len(list_k) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    # collect all k to node_list
                    node_list0 = {}
                    for k in list_k:
                        if G.nodes[k]['full'] == False:
                            node_list0[k] = G.nodes[k]['trust']
                    if len(node_list0) == 0:
                        G.nodes[j]['full'] = True
                        continue
                    node_list = {}
                    for k in node_list0.keys():
                        if edge_cur[pending.index(k)] < G.nodes[k]['edge_interaction']:
                            node_list[k] = node_list0[k]
                    if len(node_list) == 0:
                        continue

                    # rank node_list and pick top 1
                    node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
                    flag = 0
                    ranked = list(node_list.keys())  # ascending
                    length = len(ranked)

                    # select top 1 value
                    while edge_cur[idx_j] < limit_j or (edge_cur[idx_j] == limit_j and flag == 0):
                        if len(node_list) == 0 or G.nodes[j]['full'] == True:
                            break
                        ranked = list(node_list.keys())  # ascending
                        f = ranked[-1]
                        if G.nodes[f]['label'] == 'Legit' and G.nodes[j]['trust'] < 1.0 * node_list[f]: #i > 20 and
                            del node_list[f]
                            continue
                        if edge_cur[pending.index(f)] < G.nodes[f]['edge_interaction'] and G.nodes[f]['full'] == False:
                            new_friend(G, j, f)
                            flag = 1
                            edge_cur[idx_j] = edge_cur[idx_j] + 1
                            edge_cur[pending.index(f)] = edge_cur[pending.index(f)] + 1
                        del node_list[f]

            size_prev = G.size()
            full_nodes_new = set(G.nodes()) - set(friend_pending(G))
            full_nodes_iter = full_nodes_new - full_nodes

            if i < 80:
                # simulate positive interaction
                for j in list(G.nodes()):
                    if G.nodes[j]['feeding'] >= random.random():  # if feeding, share to all friends
                        for fb in list(G.neighbors(j)):  # update feeding
                            G.edges[j, fb]['f'] = G.edges[j, fb]['f'] + 1
                            if G.edges[j, fb]['f'] > G.nodes[j]['fmax']:
                                G.nodes[j]['fmax'] = G.edges[j, fb]['f']
                            if G.edges[j, fb]['f'] > G.nodes[fb]['fmax']:
                                G.nodes[fb]['fmax'] = G.edges[j, fb]['f']
                            if G.nodes[fb]['feedback'] >= random.random():  # updating feedback
                                G.edges[j, fb]['b'] = G.edges[j, fb]['b'] + 1
                                if G.edges[j, fb]['b'] > G.nodes[j]['bmax']:
                                    G.nodes[j]['bmax'] = G.edges[j, fb]['b']
                                if G.edges[j, fb]['b'] > G.nodes[fb]['bmax']:
                                    G.nodes[fb]['bmax'] = G.edges[j, fb]['b']
                # update sc and trust values after each run
                for j in list(G.nodes()):
                    social_capital_trust_update(G, j)
        else: # attack and defense
            nodes = list(G.nodes())
            random.shuffle(nodes)
            for j in nodes:
                # if j not in G.nodes():
                #     continue
                if isinstance(G.nodes[j]['phish'], int):
                    G = interaction_ATTACK(G, likers, dname, 'phish', j, p_as, records, i)  # phishing

            # stats collection
            count = 0 # reported attackers
            for j in nodes:
                if G.nodes[j]['label'] == 'Fake' and G.nodes[j]['phish'] >= 3:
                    count = count + 1
            attack_report[i-100] = count
            sir_state_change(records, attack_state, i)

            # post-attack updates: attacker report and user transformation
            for j in nodes:
                if isinstance(G.nodes[j]['phish'], int):  #attackers
                    # removed the reported attackers from the network/ compromised user back to S
                    if G.nodes[j]['phish'] >= 3:
                        if G.nodes[j]['label'] == True:  # compromised account
                            G.nodes[j]['phish'] = 'R'  # change password
                            G.nodes[j]['experience'] = G.nodes[j]['experience'] + 1
                        else:  # remove the attacker
                            nb = list(G.neighbors(j))
                            for k in nb:
                                G.remove_edge(j, k)
                else: #normal users
                    if G.nodes[j]['phish'] == 'E' and random.random() <= 0.2: #I to attacker, compromised account
                        G.nodes[j]['phish'] = 0
            sir_process_update(G, attack_sir, run, feat, i)

        # printing
        if i == 100:
            invite_times = {}
            invite_friends = {}
            for x in sorted(invite.keys()):
                if invite[x] > 0:
                    invite_times[x] = invite[x]
                    invite_friends[x] = G.nodes[x]['friends']
            print(len(invite_times))
        if (i + 1) % 20 == 0:
            print("%s edges at %d :%d" % (feature, i, G.size()))

    print((time.time() - start_time) / 60)
    sum = 0
    diff = 0
    fake_edges = {}
    fake_friends = {}
    for x in G.nodes():
        if G.nodes[x]['label'] == 'Fake':
            diff = diff + abs(G.nodes[x]['friends'] - G.nodes[x]['edges'])
            sum = sum + G.nodes[x]['friends']
            fake_edges[x] = G.nodes[x]['edges']
            fake_friends[x] = G.nodes[x]['friends']
    print("sum of fake friends:", sum)
    print("sumdiff of fake friends:", diff)
    return G

def RC(sample, capital_full, G_full, likers, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=0):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('RC:')
    G = initial_friend_SC(G, 'relational', i, dname)
    print('RC:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_SC(G, dname, likers, 'RC', p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'RC', dname, out)

def STC(sample, capital_full, G_full, likers, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=1):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('STC:')
    G = initial_friend_SC(G, 'human', i, dname)
    print('STC:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_SC(G, dname, likers, 'STC', p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'STC', dname, out)

def CC(sample, capital_full, G_full, likers, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=2):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('CC:')
    G = initial_friend_SC(G, 'cognitive', i, dname)
    print('CC:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_SC(G, dname, likers, 'CC', p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'CC', dname, out)

def SC(sample, capital_full, G_full, likers, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=3):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('SC:')
    G = initial_friend_SC(G, 'capital', i, dname)
    print('SC:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_SC(G, dname, likers, 'SC', p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'SC', dname, out)

def SA(sample, capital_full, G_full, likers, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=5):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('SA:')
    G = initial_friend_SA(G, i, dname)
    print('SA:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_SA(G, dname, likers, p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'SA', dname, out)

def TR(sample, capital_full, G_full, likers, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=4):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('TR:')
    G = initial_friend_TR(G, i, dname)
    print('TR:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_TR(G, dname, likers, 'TR', p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'TR', dname, out)

def TP(sample, capital_full, G_full, likers, topics, p_as, i, attack_sir, attack_state, attack_report, dname, out, feat=6):
    capital = capital_full.loc[sorted(sample)]
    G = nx.Graph(G_full.subgraph(sorted(sample)))
    print('TP:')
    G = initial_friend_TP(G, topics, i, dname)
    print('TP:')
    print('Full ratio:', len([n for n, v in G.nodes(data=True) if v['full'] == True]) / G.order())
    G = interaction_TP(G, dname, likers, topics, p_as, i, attack_sir, attack_state, attack_report, feat)
    capital = post_process(G, capital, i, 'TP', dname, out)

