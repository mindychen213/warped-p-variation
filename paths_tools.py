import numpy as np
from pvar_tools import *
from transformers import *
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import Parallel, delayed
import multiprocessing
import copy
from operator import itemgetter

def split(N, l):
    return [l[x:x+N] for x in range(0, len(l), N)]

class LatticePaths():
    """Takes in two piece-wise linear paths x and y as numpy arrays (lenght, dimension)
       dimension must be the same for bith x and y. length doesn't need to agree.
    """
    def __init__(self, x, y, p, depth, norm='l1', brute_force=False, parallelise=True):
        self.m = x.shape[0]
        self.n = y.shape[0]
        if len(x.shape) == 1: # use a path transform
            transformer = AddTime()
            # transformer = LeadLag()
            self.x = transformer.fit_transform([x])[0]
            self.y = transformer.fit_transform([y])[0]
        else:
            self.x = x
            self.y = y
        self.dim  = self.x.shape[1]
        self.grid = self._generate_grid() #lattice
        self.p = p
        self.depth = depth
        self.norm=norm
        if brute_force:
            self.allPaths = [] #list of all possible warping paths

            t = time.time()
            self._findPaths()
            
            print('time to find all possible paths: {0:.2f}'.format(time.time()-t))

            if parallelise:

                # Parallelisation
                num_cores = multiprocessing.cpu_count()
                self._findPaths() #call this function to generate the list allPaths

                # DP
                t = time.time()
                self.results = Parallel(n_jobs=num_cores)(delayed(self._global_warping_pvar)(l) for l in split(int(len(self.allPaths)/num_cores), self.allPaths)) #brute force with DP
                self.warped_pvar, self.best_partition, self.best_warp = min(self.results, key=itemgetter(0))
                print('total time for brute force with DP in Parallel: {0:.2f} s'.format(time.time()-t))

                # Alexey's algo
                t = time.time()
                self.optim_results = Parallel(n_jobs=num_cores)(delayed(self._optim_global_warping_pvar)(l) for l in split(int(len(self.allPaths)/num_cores), self.allPaths)) #brute force with Alexey's algo
                self.optim_warped_pvar, self.optim_best_partition, self.optim_best_warp = min(self.optim_results, key=itemgetter(0))
                print('total time for brute force with Alexeys algo in parallel: {0:.2f} s'.format(time.time()-t))

            else:
                t = time.time()
                self.warped_pvar, self.best_partition, self.best_warp = self._global_warping_pvar(self.allPaths) #brute force with DP
                print('total time for brute force with DP sequentially: {0:.2f} s'.format(time.time()-t))

                t = time.time()
                self.optim_warped_pvar, self.optim_best_partition, self.optim_best_warp = self._optim_global_warping_pvar(self.allPaths) #brute force with Alexey's algo
                print('total time for brute force with Alexey sequentiallys algo: {0:.2f} s'.format(time.time()-t))
        
        self.total_paths = self._countPaths() #total number of admissible warping paths

    def _generate_grid(self):
        """generate lattice"""
        l_outer = []
        for i in range(self.m):
            l_inner = []
            for j in range(self.n):
                l_inner.append((i,j))
            l_outer.append(l_inner)
        return l_outer

    def _countPaths(self):
        """count total number of admissible lattice paths"""
        total = [1 for k in range(self.n)]
        for i in range(self.m-1):
            for j in range(1,self.n):
                total[j] += total[j-1]
        return total[self.n-1]

    def _findPathsUtil(self, path, i, j, indx):
        """utility function to recursively build a warping path"""

        # if we reach the bottom of maze, we can only move right
        if i==self.m-1:
            for k in range(j,self.n):
                path[indx+k-j] = self.grid[i][k]
            self.allPaths.append(copy.deepcopy(path))
            return

        # if we reach to the right most corner, we can only move down
        if j == self.n-1:
            for k in range(i,self.m):
                path[indx+k-i] = self.grid[k][j]
            self.allPaths.append(copy.deepcopy(path))
            return

        path[indx] = self.grid[i][j]

        self._findPathsUtil(path, i+1, j, indx+1)
        self._findPathsUtil(path, i, j+1, indx+1)
        # @Cris to look at
        self._findPathsUtil(path, i+1, j+1, indx+1)

    def _findPaths(self):
        """Generate all admissible warping paths, i.e. lattice paths + 1-step diagonal"""
        path = [0 for d in range(self.m+self.n-1)]
        self._findPathsUtil(path, 0, 0, 0)

    def align(self, warp):
        """align x and y according to the warping path"""
        x_reparam = np.array([self.x[k] for k in [j[0] for j in warp]])
        y_reparam = np.array([self.y[k] for k in [j[1] for j in warp]])
        return x_reparam, y_reparam

    def single_warped_pvar(self, warp, p, depth, norm='l1'):
        """computes warped p-variation along one path with dynamic programming algo"""
        x_reparam, y_reparam = self.align(warp)
        pvar, partition = p_variation_distance(x_reparam, y_reparam, p=p, depth=depth, norm=norm)
        return pvar, partition

    def optim_single_warped_pvar(self, warp, p, depth, norm='l1'):
        """computes warped p-variation along one path with Alexey's algo"""
        x_reparam, y_reparam = self.align(warp)
        pvar, partition = p_variation_distance_optim(x_reparam, y_reparam, p=p, depth=depth, norm=norm)
        return pvar, partition

    def _global_warping_pvar(self, paths):
        """Brute force global warped p-variation distance with standard algo"""
        pvar_best, best_partition = self.single_warped_pvar(paths[0], self.p, self.depth, self.norm)
        best_warp = paths[0]
        for w in paths[1:]:
            pvar, partition = self.single_warped_pvar(w, self.p, self.depth, self.norm)
            if pvar < pvar_best:
                pvar_best = pvar
                best_partition = partition
                best_warp = w
        return pvar_best, best_partition, best_warp

    def _optim_global_warping_pvar(self, paths):
        """Brute force global warped p-variation distance with Alexey's algo"""
        pvar_best, best_partition = self.optim_single_warped_pvar(paths[0], self.p, self.depth, self.norm)
        best_warp = paths[0]
        for w in paths[1:]:
            pvar, partition = self.optim_single_warped_pvar(w, self.p, self.depth, self.norm)
            if pvar < pvar_best:
                pvar_best = pvar
                best_partition = partition
                best_warp = w
        return pvar_best, best_partition, best_warp

    # plotting @Cris TODO(think about multidimensional paths)
    def plot_alignment(self):
        plt.plot(self.x.T[1], 'bo-' ,label = 'x')
        plt.plot(self.y.T[1], 'g^-', label = 'y')
        plt.title('Alignment')
        plt.legend()
        for (map_x, map_y) in self.best_warp:
            plt.plot([map_x, map_y], [self.x.T[1][map_x], self.y.T[1][map_y]], 'r')