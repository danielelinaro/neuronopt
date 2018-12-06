#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pickle

pop = pickle.load(open('final_population.pkl','r'))
evaluator = pickle.load(open('evaluator.pkl','r'))

with open('final_population.csv','w') as fid:
    for i,name in enumerate(evaluator.param_names):
        fid.write('%s' % name)
        for p in [indiv[i] for indiv in pop]:
            fid.write(',%f' % p)
        fid.write('\n')

N_indiv = len(pop)
N_pars = len(pop[0])

plt.figure()

for i in range(N_pars):
    Y = np.array([indiv[i] for indiv in pop])
    X = np.random.uniform(size=N_indiv)*0.2 + i
    m = np.median(Y)
    plt.semilogy(X,Y,'k.')
    plt.plot([i,i+0.2],[m,m],'r',lw=2)

plt.show()
