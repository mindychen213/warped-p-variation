import sys
sys.path.insert(0, '../backend')
sys.path.insert(0, '../../data')

import numpy as np
import pybnb
import time
import matplotlib.pyplot as plt

from branch_and_bound_warped_pvar import BnBWarping
from transformers import *

idx = np.linspace(0, 4*np.pi, 15)

x = np.sin(idx)
y = np.cos(idx)

x = AddTime().fit_transform([x])[0]
y = AddTime().fit_transform([y])[0]

problem = BnBWarping(x, y, depth=2, norm='l1', p=1.05, root_node=(0, 0), bc=1, 
                     plot_2d=False, pvar_advanced=False, pvar_dist_mem=None)

solver = pybnb.Solver()

print(f'\n BnB algo is using {solver.worker_count} core/s')

results = solver.solve(problem, log=None, queue_strategy='depth')

print('warped p-var: {:.2f}'.format(results.objective))
print('wall time: {:.2f} secs \n \n'.format(results.wall_time))

#best_warp = results.best_node.state
#_, optimal_partition = problem.distance(results.best_node.state, optim_partition=True)
#problem.plot_alignment([best_warp[k] for k in optimal_partition])
#plt.show()





