import numpy as np
from pvar_tools import *

class LatticePaths():
    """Takes in two piece-wise linear paths x and y as numpy arrays (lenght, dimension)
       dimension must be the same for bith x and y. length doesn't need to agree.
    """
    def __init__(self, x, y, p, depth, norm='l1', brute_force=False):
        self.x = x
        self.y = y
        self.m = x.shape[0]
        self.n = y.shape[0]
        self.dim  = x.shape[1]
        self.grid = self._generate_grid() #lattice
        self.p = p
        self.depth = depth
        self.norm=norm
        if brute_force:
            self.allPaths = [] #list of all possible warping paths
            self._findPaths() #call this function to generate the list allPaths
            self.warped_pvar = self._global_warping_pvar(p, depth, norm) #brute force with DP
            self.optim_warped_pvar = self._optim_global_warping_pvar(p, depth, norm) #brute force with Alexey's algo
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
            self.allPaths.append(path)
            return

        # if we reach to the right most corner, we can only move down
        if j == self.n-1:
            for k in range(i,self.m):
                path[indx+k-i] = self.grid[k][j]
            self.allPaths.append(path)
            return

        path[indx] = self.grid[i][j]

        self._findPathsUtil(path, i+1, j, indx+1)
        self._findPathsUtil(path, i, j+1, indx+1)
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
        return p_variation_distance(x_reparam, y_reparam, p=p, depth=depth, norm=norm)

    def optim_single_warped_pvar(self, warp, p, depth, norm='l1'):
        """computes warped p-variation along one path with Alexey's algo"""
        x_reparam, y_reparam = self.align(warp)
        pvar, partition = p_variation_distance_optim(x_reparam, y_reparam, p=p, depth=depth, norm=norm)
        return pvar, partition

    def _global_warping_pvar(self, p, depth, norm='l1'):
        """Brute force global warped p-variation distance with standard algo"""
        pvars = []
        for w in self.allPaths:
            pvars.append(self.single_warped_pvar(w, p, depth, norm))
        return min(pvars)

    def _optim_global_warping_pvar(self, p, depth, norm='l1'):
        """Brute force global warped p-variation distance with Alexey's algo"""
        pvars = []
        for w in self.allPaths:
            pvars.append(self.optim_single_warped_pvar(w, p, depth, norm))
        return min(pvars)