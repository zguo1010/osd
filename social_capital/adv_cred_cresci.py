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
from numpy.random import randint, choice, seed
from datetime import date
import sys
from multiprocessing import Process, Value, Array
from adv_cred_functions import *
plt.switch_backend('agg')


'''calculating social capital in three dimensions '''
def initialization_capital(likers, mu, su):
    HC = likers['age'] / max(likers['age']) + likers['lines'] / max(likers['lines'])
    HC = 3 * HC / max(HC)
    # HC.head()   'num_hashtags'， 'ave_comment' too many zeros
    CC = likers['category'] / max(likers['category']) + likers['num_urls'] / max(likers['num_urls']) + likers[
        'num_mentions'] / max(likers['num_mentions']) + likers['ave_share'] / max(likers['ave_share'])
    CC = 3 * CC / max(CC)
    # CC.head()
    RC = likers['freq_posts'] / max(likers['freq_posts']) + likers['favorite_tweets'] / max(likers['favorite_tweets']) + \
         likers['freq_replies'] / max(likers['freq_replies'])
    RC = 3 * RC / max(RC)
    capital = pd.concat([likers['label'], HC], axis=1)
    capital.columns = ["label", "human"]
    capital["cognitive"] = CC
    capital["relational"] = RC

    # behavioral seeds, fixed, by label
    ld = capital['label'] == 'Legit'
    fd = capital['label'] == 'Fake'
    capital.loc[ld, 'feeding'] = likers.loc[ld, 'freq_posts'] / max(likers.loc[ld, 'freq_posts'])
    capital.loc[fd, 'feeding'] = likers.loc[fd, 'freq_posts'] / max(likers.loc[fd, 'freq_posts'])
    capital.loc[ld, 'posting'] = likers['num_urls'] / max(likers['num_urls']) #legit user post URLs
    # capital['feeding'] = likers['ave_post']/max(likers['ave_post'])
    capital.loc[ld, 'feedback'] = likers.loc[ld, 'freq_replies'] / max(likers.loc[ld, 'freq_replies'])
    capital.loc[fd, 'feedback'] = likers.loc[fd, 'freq_replies'] / max(likers.loc[fd, 'freq_replies'])
    # capital['feedback'] = likers['ave_like']/max(likers['ave_like'])
    capital.loc[ld, 'inviting'] = likers.loc[ld, 'friends'] / np.percentile(likers.loc[ld, 'friends'], 92)
    capital.loc[fd, 'inviting'] = likers.loc[fd, 'friends'] / np.percentile(likers.loc[fd, 'friends'], 99)
    # capital['inviting'] = likers['friends'] / max(likers['friends'])
    ### competence of normal user, c=-lambda*(follower-f_mean)**2+3
    ave_competence = np.percentile(likers.loc[ld, 'followers'], 50)
    lamb = 2/(1 - ave_competence)**2
    capital['competence'] = 3 - lamb * np.square(likers['followers'] - ave_competence)
    capital.loc[capital['competence'] < 1, 'competence'] = 1 #P^crd seems too low
    capital["deception"] = 3  # default deception quality
    capital.loc[fd, 'deception'] = np.random.normal(mu, su, len(capital.loc[fd, 'deception'])) #deception quality ranges in [0.5,3]
    capital.loc[capital['deception'] > 3, 'deception'] = 3
    capital.loc[capital['deception'] < 1, 'deception'] = 1

    # other attributes
    capital['edges'] = 0
    capital["STC"] = 0.0
    capital["CC"] = 0.0
    capital["RC"] = 0.0
    capital["SC"] = 0.0
    return capital


def main(argv):
    #parameters
    dname = "Cresci15"
    low = int(argv[0]) #low/high range
    n = 1946 #total number of nodes
    p_as = 0.1 #attack target ratio/% of friends
    p = float(argv[1]) #p_a, Fake user ratio
    attack = 105
    run = 100 #simulation runs
    mu = float(argv[2])
    su = float(argv[3])

    #output folder
    if low == 90:
        low = 80
        out = str(int(p * 100))
    else:
        out = str(low)

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
    likers.loc[likers['followers']>500, 'followers'] = 500
    likers.loc[likers['followers']==0, 'followers'] = 1

    # reduce friend number by scale
    scale = 0.08
    likers['friends'] = np.ceil(likers['friends'] * scale)

    # normalize large values
    maximum_cap(likers, 'lines', 0.05)#25
    likers.loc[likers['ave_comment']>1, 'ave_comment'] = 1 #133
    likers.loc[likers['ave_share']>500, 'ave_share'] = 500 #525
    maximum_cap(likers, 'category', 0.05)#56
    maximum_cap(likers, 'num_urls', 0.03)#3
    maximum_cap(likers, 'num_mentions', 0.05)#61
    likers.loc[likers['favorite_tweets']>300, 'favorite_tweets'] = 300 #317
    maximum_cap(likers, 'freq_posts', 0.1)#370/65
    maximum_cap(likers, 'freq_replies', 0.1)#268/67

    # prepare data
    print('friend max: ', max(likers['friends']))
    f = int(np.percentile(likers['friends'], 90))
    print('friend 90%:', f)
    attack_sir = {x1: {'phish':{x3:np.zeros((run,7)) for x3 in ['s', 'i', 'r', 'e']}} for x1 in range(100, attack)}
    attack_std = {x1: {'phish':{x3:np.zeros((run,7)) for x3 in ['s', 'i', 'r', 'e']}} for x1 in range(100, attack)}
    attack_state = {x1: {x3: np.zeros((run, 7)) for x3 in ['ss', 'sr', 'ir', 'sir']} for x1 in range(100, attack)}
    attack_report = {x1: np.zeros((run, 7)) for x1 in range(100, attack)}
    attack_stdrep = {x1: np.zeros((run, 7)) for x1 in range(100, attack)}

    # prepare full graph and capital dataframes
    likers_full = likers
    topics_full = topics
    capital_full = initialization_capital(likers_full, mu, su)
    G_full = initialization_graph(likers_full, capital_full, full_size, dname)

    '''# drawing full graph: plot SC HC CC for 2464 nodes
    fitting_distribution(capital_full, 'human', dname, 'All nodes', True, 50, out)
    fitting_distribution(capital_full, 'cognitive', dname, 'All nodes', True, 50, out)
    fitting_distribution(capital_full, 'relational', dname, 'All nodes', True, 50, out)
    print(QG)'''

    # fix seed for low/high experiment
    seed_i = (date(2021, 4, 28) - date(2020, 7, 1)).days
    seed(seed_i)
    print('seed: ', seed_i, 'p: ', p, 'mu: ', mu, "su: ", su)
    # Fake users
    fake_num = int(n * p)
    fake_data = likers_full.loc[likers_full['label'] == 'Fake', :]
    fake_sample = choice(fake_data.index, fake_num, replace=False)
    # Legit users
    legit_num = n
    legit_data = likers_full.loc[likers_full['label'] == 'Legit', :]
    legit_sample = choice(legit_data.index, legit_num, replace=False)
    sample = np.concatenate((fake_sample, legit_sample))
    print(sorted(sample)[:20])
    likers_s = likers_full.loc[sorted(sample)]
    topics_s = topics_full.loc[sorted(sample)]


    #start simulation runs
    start_time_i = time.time()
    directory = 'resultcres/' + out + '/'
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if file.endswith(".txt")]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)
    for i in range(run):
        # Shared memory variables for Process each interaction
        sir_rc, sir_cc = Array('d', range(4*(attack-100))), Array('d', range(4*(attack-100)))
        sir_stc, sir_sc = Array('d', range(4*(attack-100))), Array('d', range(4*(attack-100)))
        sir_sa, sir_tp = Array('d', range(4 * (attack - 100))), Array('d', range(4 * (attack - 100)))
        sir_tr, cnt_tr = Array('d', range(4*(attack-100))), Array('d', range(4 * (attack - 100)))
        cnt_rc, cnt_cc = Array('i', range(4 * (attack - 100))), Array('i', range(4 * (attack - 100)))
        cnt_stc, cnt_sc = Array('i', range(4 * (attack - 100))), Array('i', range(4 * (attack - 100)))
        cnt_sa, cnt_tp = Array('i', range(4 * (attack - 100))), Array('i', range(4 * (attack - 100)))
        rep_rc, rep_cc = Array('i', range(attack - 100)), Array('i', range(attack - 100))
        rep_stc, rep_sc = Array('i', range(attack - 100)), Array('i', range(attack - 100))
        rep_sa, rep_tp = Array('i', range(attack - 100)), Array('i', range(attack - 100))
        rep_tr = Array('i', range(attack - 100))

    #running RC friend
        threads = list()
        x2=Process(target=RC, args=(sample, capital_full, G_full, likers_s, p_as, i, sir_rc, cnt_rc, rep_rc, dname, out))
        threads.append(x2)
        x2.start()

    #running STC friend
        x3=Process(target=STC, args=(sample, capital_full, G_full, likers_s, p_as, i, sir_stc, cnt_stc, rep_stc, dname, out))
        threads.append(x3)
        x3.start()

    #running CC friend
        x4 = Process(target=CC, args=(sample, capital_full, G_full, likers_s, p_as, i, sir_cc, cnt_cc, rep_cc, dname, out))
        threads.append(x4)
        x4.start()

    #running SC friend
        x5 = Process(target=SC, args=(sample, capital_full, G_full, likers_s, p_as, i, sir_sc, cnt_sc, rep_sc, dname, out))
        threads.append(x5)
        x5.start()

    # running SA friend
        x6 = Process(target=SA, args=(sample, capital_full, G_full, likers_s, p_as, i, sir_sa, cnt_sa, rep_sa, dname, out))
        threads.append(x6)
        x6.start()
    #     x6.join()
    #running Trust friend
        x7 = Process(target=TR, args=(sample, capital_full, G_full, likers_s, p_as, i, sir_tr, cnt_tr, rep_tr, dname, out))
        threads.append(x7)
        x7.start()
        # x7.join()
    #running Topics friend
        x8 = Process(target=TP, args=(sample, capital_full, G_full, likers_s, topics_s, p_as, i, sir_tp, cnt_tp, rep_tp,
                    dname, out))
        threads.append(x8)
        x8.start()
        # x8.join()

        # print info for each iteration
        for thread in threads:
            thread.join()
        print('Iteration ',i,' total time: ', (time.time()-start_time_i)/60)

        # update the stats for each iteration
        for j in range(100, attack):
            attack_sir[j]['phish']['s'][i][0], attack_sir[j]['phish']['i'][i][0], attack_sir[j]['phish']['r'][i][0], \
            attack_sir[j]['phish']['e'][i][0] = sir_rc[(j - 100) * 4: (j - 99) * 4]
            attack_sir[j]['phish']['s'][i][1], attack_sir[j]['phish']['i'][i][1], attack_sir[j]['phish']['r'][i][1], \
            attack_sir[j]['phish']['e'][i][1] = sir_stc[(j - 100) * 4: (j - 99) * 4]
            attack_sir[j]['phish']['s'][i][2], attack_sir[j]['phish']['i'][i][2], attack_sir[j]['phish']['r'][i][2], \
            attack_sir[j]['phish']['e'][i][2] = sir_cc[(j - 100) * 4: (j - 99) * 4]
            attack_sir[j]['phish']['s'][i][3], attack_sir[j]['phish']['i'][i][3], attack_sir[j]['phish']['r'][i][3], \
            attack_sir[j]['phish']['e'][i][3] = sir_sc[(j - 100) * 4: (j - 99) * 4]
            attack_sir[j]['phish']['s'][i][4], attack_sir[j]['phish']['i'][i][4], attack_sir[j]['phish']['r'][i][4], \
            attack_sir[j]['phish']['e'][i][4] = sir_tr[(j - 100) * 4: (j - 99) * 4]
            attack_sir[j]['phish']['s'][i][5], attack_sir[j]['phish']['i'][i][5], attack_sir[j]['phish']['r'][i][5], \
            attack_sir[j]['phish']['e'][i][5] = sir_sa[(j - 100) * 4: (j - 99) * 4]
            attack_sir[j]['phish']['s'][i][6], attack_sir[j]['phish']['i'][i][6], attack_sir[j]['phish']['r'][i][6], \
            attack_sir[j]['phish']['e'][i][6] = sir_tp[(j - 100) * 4: (j - 99) * 4]

            attack_state[j]['ss'][i][0], attack_state[j]['sr'][i][0], attack_state[j]['ir'][i][0], \
            attack_state[j]['sir'][i][0], = cnt_rc[(j - 100) * 4: (j - 99) * 4]
            attack_state[j]['ss'][i][1], attack_state[j]['sr'][i][1], attack_state[j]['ir'][i][1], \
            attack_state[j]['sir'][i][1], = cnt_stc[(j - 100) * 4: (j - 99) * 4]
            attack_state[j]['ss'][i][2], attack_state[j]['sr'][i][2], attack_state[j]['ir'][i][2], \
            attack_state[j]['sir'][i][2], = cnt_cc[(j - 100) * 4: (j - 99) * 4]
            attack_state[j]['ss'][i][3], attack_state[j]['sr'][i][3], attack_state[j]['ir'][i][3], \
            attack_state[j]['sir'][i][3], = cnt_sc[(j - 100) * 4: (j - 99) * 4]
            attack_state[j]['ss'][i][4], attack_state[j]['sr'][i][4], attack_state[j]['ir'][i][4], \
            attack_state[j]['sir'][i][4], = cnt_tr[(j - 100) * 4: (j - 99) * 4]
            attack_state[j]['ss'][i][5], attack_state[j]['sr'][i][5], attack_state[j]['ir'][i][5], \
            attack_state[j]['sir'][i][5], = cnt_sa[(j - 100) * 4: (j - 99) * 4]
            attack_state[j]['ss'][i][6], attack_state[j]['sr'][i][6], attack_state[j]['ir'][i][6], \
            attack_state[j]['sir'][i][6], = cnt_tp[(j - 100) * 4: (j - 99) * 4]

            attack_report[j][i][0], attack_report[j][i][1], attack_report[j][i][2] \
                = rep_rc[j - 100], rep_stc[j - 100], rep_cc[j - 100]
            attack_report[j][i][3], attack_report[j][i][4], attack_report[j][i][5], attack_report[j][i][6] \
                = rep_sc[j - 100], rep_tr[j - 100], rep_sa[j - 100], rep_tp[j - 100]

    #outside i iterations/ calculate mean and std for repeats and 6 conditions
    for j in range(100, attack):
        attack_std[j]['phish']['s'] = np.std(attack_sir[j]['phish']['s'], axis = 0)
        attack_std[j]['phish']['i'] = np.std(attack_sir[j]['phish']['i'], axis = 0)
        attack_std[j]['phish']['r'] = np.std(attack_sir[j]['phish']['r'], axis = 0)
        attack_std[j]['phish']['e'] = np.std(attack_sir[j]['phish']['e'], axis = 0)

        attack_sir[j]['phish']['s'] = np.mean(attack_sir[j]['phish']['s'], axis = 0)
        attack_sir[j]['phish']['i'] = np.mean(attack_sir[j]['phish']['i'], axis = 0)
        attack_sir[j]['phish']['r'] = np.mean(attack_sir[j]['phish']['r'], axis = 0)
        attack_sir[j]['phish']['e'] = np.mean(attack_sir[j]['phish']['e'], axis = 0)

        attack_state[j]['ss'] = np.mean(attack_state[j]['ss'], axis=0)
        attack_state[j]['sr'] = np.mean(attack_state[j]['sr'], axis=0)
        attack_state[j]['ir'] = np.mean(attack_state[j]['ir'], axis=0)
        attack_state[j]['sir'] = np.mean(attack_state[j]['sir'], axis=0)

        attack_stdrep[j] = np.std(attack_report[j], axis=0)
        attack_report[j] = np.mean(attack_report[j], axis=0)

    print(attack_report)
    print(attack_sir)
    print(attack_state)
    print(attack_std)

    #save huge results into files
    attack_list = [list(attack_sir[x1][x2][x3])
                   for x1 in range(100, attack)
                   for x2 in ['phish']
                   for x3 in ['s', 'i', 'r', 'e']]

    std_list = [list(attack_std[x1][x2][x3])
                   for x1 in range(100, attack)
                   for x2 in ['phish']
                   for x3 in ['s', 'i', 'r', 'e']]

    state_list = [list(attack_state[x1][x3])
                   for x1 in range(100, attack)
                   for x3 in ['ss', 'sr', 'ir', 'sir']]

    # save small results into files
    with open('sir_c_'+ out +'.txt', 'w') as f:
        for item in attack_list:
            f.write("%s\n" % item)

    with open('std_c_'+ out +'.txt', 'w') as f:
        for item in std_list:
            f.write("%s\n" % item)

    with open('state_c_'+ out +'.txt', 'w') as f:
        for item in state_list:
            f.write("%s\n" % item)

    with open('stats_c_'+ out +'.txt', 'w') as f:
        for x1 in range(100, attack):
            f.write("%s\n" % list(attack_report[x1]))
        for x1 in range(100, attack):
            f.write("%s\n" % list(attack_stdrep[x1]))

if __name__ == '__main__':
    main(sys.argv[1:])