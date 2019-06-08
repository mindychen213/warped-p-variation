import sys
sys.path.insert(0, '../backend')
sys.path.insert(0, '../../data')

#flag = sys.argv[1]
#size = sys.argv[2]

#if flag == '1':
#    print('using tight bound sequentially \n')
#    use_dp = True
#else:
#    print('using soft bound in parallel \n')
#    use_dp = False

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

problem = BnBWarping(x, y, depth=2, norm='l1', p=1.5, root_node=(0, 0),
                     plot_2d=False, boundary_condition=4, initialize_memoization=True,
                     use_dp=True, with_sig_memoization=True, pvar_advanced=False, pth_root=True)

solver = pybnb.Solver()

print(f'\n BnB algo is using {solver.worker_count} cores')

results = solver.solve(problem, log=None, queue_strategy='depth')

print('warped p-var: {:.2f}'.format(results.objective))
print('wall time: {:.2f} secs \n \n'.format(results.wall_time))

#best_warp = results.best_node.state
#_, optimal_partition = problem.distance(results.best_node.state, optim_partition=True)
#problem.plot_alignment([best_warp[k] for k in optimal_partition])
#plt.show()





