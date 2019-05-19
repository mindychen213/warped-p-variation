import sys
sys.path.insert(0, '../backend')
sys.path.insert(0, '../../data')

import numpy as np
import pybnb
import time
import matplotlib.pyplot as plt

from branch_and_bound_warped_pvar import BnBWarping

from transformers import *

idx = np.linspace(0, 6.28, 10)
x = np.sin(idx)
y = np.cos(idx)

x = AddTime().fit_transform([x])[0]
y = AddTime().fit_transform([y])[0]

problem = BnBWarping(x, y, depth=2, norm='l1', p=1.5)

# solving problem sequentially
solver_seq = pybnb.Solver(comm=None)

# solving problem in parallel
solver_par = pybnb.Solver()

results_seq = solver_seq.solve(problem, log=None, 
                               queue_strategy='depth')

results_par = solver_par.solve(problem, log=None, 
                               queue_strategy='depth')

assert np.abs(results_seq.objective - results_par.objective) < 1e-8

print('sequential stats: ', solver_seq.collect_worker_statistics())
print('sequential stats: ', solver_par.collect_worker_statistics())



