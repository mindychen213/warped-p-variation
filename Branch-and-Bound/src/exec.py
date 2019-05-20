import sys
sys.path.insert(0, '../backend')
sys.path.insert(0, '../../data')

import numpy as np
import pybnb
import time
import matplotlib.pyplot as plt

from branch_and_bound_warped_pvar import BnBWarping

from transformers import *

idx = np.linspace(0, 6.28, 5)
x = np.sin(idx)
y = np.cos(idx)

x = AddTime().fit_transform([x])[0]
y = AddTime().fit_transform([y])[0]

problem = BnBWarping(x, y, depth=2, norm='l1', p=1.5)

solver = pybnb.Solver()

print(solver.worker_count)

results = solver.solve(problem, log=None, queue_strategy='depth')

print('warped p-var: {:.2f}'.format(results.objective))
print('wall time: {:.2f} secs'.format(results.wall_time))





