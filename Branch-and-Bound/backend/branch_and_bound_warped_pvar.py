import sys
sys.path.insert(0, '../data')

import pybnb
import numpy as np
import math

from pvar_tools import *
import matplotlib.pyplot as plt

class BnBWarping(pybnb.Problem):
    
    """ The solver in pybnb keeps track of the best solution seen so far for you, 
        and will prune the search space by not calling the branch() method when it 
        encounters a node whose bound() is worse than the best objective() seen so far.
    """

    def __init__(self, x, y, p, depth, norm='l1', plot_2d=False, boundary_condition=1, use_dp=True):

        self.x = np.array(x)
        self.y = np.array(y)

        self.m = len(self.x)
        self.n = len(self.y)
        assert self.m > 0
        assert self.n > 0

        self.plot_2d = plot_2d # flag to plot paths in 1-d or 2-d case

        self.use_dp = use_dp # flag for enable usage of dynamic programming logic
        # dynamic programming is only allowed to kick in if we are at least 
        # boundary_conditions nodes away from root.
        self.boundary_condition = boundary_condition

        self.p = p # p for p-variation
        self.depth = depth # signature depth
        self.norm = norm # l1 or l2 norm

        self.path = [(0,0), (0,0)] # lattice path

        self.sig_memoizer = {(0,0):0.} # keep track of the history of signatures of sub-pathlets
        self.bound_memoizer = {} # keep track of the history of bound function evaluations
        self.best_node_value = math.inf # keep track of best bound

        #self.count_soft_bound_calls = 0
        #self.count_soft_bound_sufficiency = 0
        #self.count_tight_bound_calls = 0
        #self.count_tight_bound_sufficiency = 0
        #self.all_the_rest = 0

    def align(self, warp):
        """align x and y according to the warping path"""
        x_reparam = np.array([self.x[k] for k in [j[0] for j in warp]])
        y_reparam = np.array([self.y[k] for k in [j[1] for j in warp]])
        return x_reparam, y_reparam

    def pairwise_sig_norm(self, path1, path2, a, b):
        """compute ||S(path1)_ab - S(path2)_ab||"""
        if (a,b) in self.sig_memoizer:
            sig_x, sig_y = self.sig_memoizer[(a,b)]
        else:
            sig_x = signature(path1[a:b+1,:], self.depth)
            sig_y = signature(path2[a:b+1,:], self.depth)
            self.sig_memoizer[(a,b)] = (sig_x, sig_y)
        return sig_norm(sig_x - sig_y, self.norm)

    def p_variation_distance(self, path1, path2, optim_partition):
        """path1 and path2 must be numpy arrays of equal lenght.
	    return the signature p-variation distance and points of the optimal partition""" 
        assert len(path1) == len(path2)
        length = len(path1)
        dist = lambda a,b: self.pairwise_sig_norm(path1, path2, a, b)
        return p_var_backbone_ref(length, self.p, dist, optim_partition)

    def distance(self, warp, optim_partition=False):
        """computes warped p-variation along one path with dynamic programming algo"""
        x_reparam, y_reparam = self.align(warp)
        pvar, partition = self.p_variation_distance(x_reparam, y_reparam, 
                                                    optim_partition=optim_partition)
        #pvar, partition = p_variation_distance(x_reparam, y_reparam, p=self.p, depth=self.depth, 
        #                                       norm=self.norm, optim_partition=optim_partition)
        return pvar, partition
        
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

    def bound(self):
        """ This function is evaluate at a partial path and needs to be a lower bound on any complete 
            path originating from it, so it can decide if the search needs to continue 
            along a partial path based on the best known objective.
        """
        
        #return self.unbounded_objective()

        a = 0. # quantity to be added to soft bound to form tight bound (i.e. solution to sub-problem)
        b, _ = self.distance(self.path) # warped p-variation distance along path so far (soft bound)
        i,j = self.path[-1] # current position in lattice

        #self.count_soft_bound_calls += 1

        if self.use_dp: # use tight bound (dynamic programming within BnB) if necessary

            # if the soft bound is not enough then compute the tight bound
            if b < self.best_node_value: 

                #self.count_tight_bound_calls += 1 

                # Boundary condition: use tight bound only if we are at least
                # boundary_condition nodes away from root
                if (i<self.boundary_condition) or (j<self.boundary_condition):
                    a = 0.
                else:
                    # Bound memoization
                    if (i,j) in self.bound_memoizer:
                        a = self.bound_memoizer[(i,j)]
                    else:
                        sub_x = self.x[i:]
                        sub_y = self.y[j:]
                        sub_problem = BnBWarping(x=sub_x, y=sub_y, p=self.p, depth=self.depth, norm=self.norm, 
                                                 boundary_condition=self.boundary_condition, use_dp=self.use_dp)
                        results = pybnb.Solver().solve(sub_problem, log=None, queue_strategy='depth')
                        a = results.objective
                        self.bound_memoizer[(i,j)] = copy.deepcopy(a)

                #if a + b < self.best_node_value:
                #    self.count_tight_bound_sufficiency += 1 
                #else:
                #    self.all_the_rest += 1

            #else: # in this case the soft bound is enough to reject
            #    self.count_soft_bound_sufficiency += 1

        return b + a

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
            
    def notify_new_best_node(self, node, current):   
        self.best_node_value = node.objective

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