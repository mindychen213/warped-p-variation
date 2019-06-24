import sys
sys.path.insert(0, '../data')

import pybnb
import numpy as np
import math
import copy
import time
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import defaultdict

import pvar_backend
import pvar_tools 

class BnBWarping(pybnb.Problem):
    
    """ The solver in pybnb keeps track of the best solution seen so far for you, 
        and will prune the search space by not calling the branch() method when it 
        encounters a node whose bound() is worse than the best objective() seen so far.

        ##################################################################################
                            
                            Parallel computations using dyadic signatures

        ##################################################################################

    """

    def __init__(self, x, y, p, depth, norm='l1', root_node=(0,0), bc=4, plot_2d=True):

        """Inputs:
                   - x, y: input paths
                   - p: p for p-variation
                   - depth: signature truncation depth
                   - norm: norm for pairwise signature distance 
                   - root_node: node at the start of the tree
                   - bc: boundary conditon for starting using the tight bound
                   - plot_2d: whether to plot results of 1-d or 2-d paths
        """

        # input paths
        self.x = np.array(x)
        self.y = np.array(y)
        self.m = len(self.x)
        self.n = len(self.y)
        assert self.m > 0
        assert self.n > 0

        self.p = p # p for p-variation
        self.depth = depth # signature depth
        self.norm = norm # l1 or l2 norm

        self.path = [(0,0), (0,0)] # lattice path
        self.best_node_value = math.inf # keep track of best bound
        self.i0, self.j0 = root_node # tuple of indeces of root node

        self.plot_2d = plot_2d # flag to plot paths in 1-d or 2-d case

        # boundary condition, i.e. necessary min distance from root node to start using tight bound
        self.bc = bc 

    def Delannoy_number(self, m, n):
        """Returns number number of paths from the southwest corner (0, 0) of a rectangular 
           grid to the northeast corner (m, n), using only single steps north, northeast, or east
        """
        if (m == 0 or n == 0): 
            return 1
        return self.Delannoy_number(m-1, n) + self.Delannoy_number(m-1, n-1) + self.Delannoy_number(m,n-1)

    #@lru_cache(maxsize=1024)
    def signature_x(self, I, J):
        i_0 = I - self.i0
        i_N = J - self.i0
        return pvar_tools.signature(self.x[i_0:i_N+1], self.depth)

    #@lru_cache(maxsize=1024)
    def signature_y(self, I, J):
        j_0 = I - self.j0
        j_N = J - self.j0
        return pvar_tools.signature(self.y[j_0:j_N+1], self.depth)

    #@lru_cache(maxsize=1024)
    def signature_norm_diff(self, i, j, I, J):
        sig_x = self.signature_x(i, I)
        sig_y = self.signature_y(j, J)
        return pvar_tools.sig_norm(sig_x, sig_y, self.norm)

    def projections_warp2paths(self, warp):
        """Given a warping path in the lattice returns:
           1) index_x_reparam: time parametrization driven by warp of the input path x
           2) index_y_reparam: time parametrization driven by warp of the input path y
        """
        index_x_reparam = []
        index_y_reparam = []
        for i,j in warp:
            index_x_reparam.append(i)
            index_y_reparam.append(j)
        return index_x_reparam, index_y_reparam

    def distance(self, warp, optim_partition=False):
        """computes warped p-variation along one path with dynamic programming algo"""

        length = len(warp)
        index_x_reparam, index_y_reparam = self.projections_warp2paths(warp)

        def dist(a, b):
            i_0, i_N = index_x_reparam[a], index_x_reparam[b]
            j_0, j_N = index_y_reparam[a], index_y_reparam[b]
            return self.signature_norm_diff(i_0+self.i0, 
                                            j_0+self.j0, 
                                            i_N+self.i0, 
                                            j_N+self.j0)
            
        res = pvar_backend.p_var_backbone_ref(length, self.p, dist, optim_partition)
        return res        
           
    def sense(self):
        return pybnb.minimize

    def objective(self):
        """ The search space is not all paths in the tree, but only complete paths, 
            i.e. paths terminating at (m,n), the very last node for all branches.
            by returning self.distance(self.path) only when self.path is a complete 
            path will ensure to optimise over the right search space (instead of 
            optimising over all possible partial paths on the tree).
        """

        if self.path[-1] == (self.m-1,self.n-1):
            val, _ = self.distance(self.path)
        else:
            val = self.infeasible_objective()

        return val

    def bound1(self, warp):
        """inf_w(d_pvar(x \circ w_x, y \circ w_y)) >= ||S(x \circ w_x) - S(y \circ w_y)||"""

        i, j = warp[0]
        I, J = warp[-1]

        return self.signature_norm_diff(i+self.i0, 
                                        j+self.j0, 
                                        I+self.i0, 
                                        J+self.j0)

    def bound2(self, warp):
        """warped p-variation distance along path so far"""
        b, _ = self.distance(warp)
        return b

    #@lru_cache(maxsize=None)
    def bound3_precomputation(self, I, J):

        i = I - self.i0
        j = J - self.j0

        if (i>self.bc) and (j>self.bc): 

            sub_x = self.x[i:]
            sub_y = self.y[j:]

            sub_problem = BnBWarping(x=sub_x, y=sub_y, p=self.p, depth=self.depth, norm=self.norm, 
                                     root_node=(i,j), bc=1, plot_2d=self.plot_2d)

            return pybnb.Solver(comm=None).solve(sub_problem, log=None, queue_strategy='depth').objective

        return 0.

    def bound3(self, warp):
        """Dynamic programming bound (using solution to sub-problems)"""
        i,j = warp[-1] # current position in lattice
        return self.bound3_precomputation(i+self.i0,j+self.j0)

    def compute_bound(self, warp):
        # Cascading better and better bounds (when necessary)

        b = self.bound1(warp)

        if b < self.best_node_value:
            b = self.bound2(warp)
            if b < self.best_node_value:
                b = (b**self.p + self.bound3(warp)**self.p)**(1./self.p)

        return b

    def bound(self):
        """ This function is evaluated at a partial path and needs to be a lower bound on any complete 
            path originating from it, so it can decide if the search needs to continue 
            along a partial path based on the best known objective.
        """
        return self.compute_bound(self.path)
        
    def notify_new_best_node(self, node, current):   
        self.best_node_value = node.objective

    def save_state(self, node):
        node.state = list(self.path)

    def load_state(self, node):
        self.path = node.state

    def branch(self):
        
        i,j = self.path[-1]
        
        if (i==self.m-1) and (j<self.n-1):
            child = pybnb.Node()
            child.state = self.path + [(i,j+1)]
            yield child
        
        elif (i<self.m-1) and (j==self.n-1):
            child = pybnb.Node()
            child.state = self.path + [(i+1,j)]
            yield child
        
        elif (i<self.m-1) and (j<self.n-1):
            nodes_update = [(i+1,j+1), (i,j+1), (i+1,j)]
            for v in nodes_update:
                child = pybnb.Node()
                child.state = self.path + [v]
                yield child

    def plot_alignment(self, best_warp):
        if self.plot_2d:
            plt.plot(self.x.T[0], self.x.T[1], 'bo-' ,label = 'x')
            plt.plot(self.y.T[0], self.y.T[1], 'g^-', label = 'y')
        else:
            plt.plot(self.x, 'bo-' ,label = 'x')
            plt.plot(self.y, 'g^-', label = 'y')
        plt.title('Alignment')
        plt.legend()
        for map_x, map_y in best_warp:
            if self.plot_2d:
                plt.plot([self.x[map_x][0], self.y[map_y][0]], [self.x[map_x][1], self.y[map_y][1]], 'r')
            else:
                plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')