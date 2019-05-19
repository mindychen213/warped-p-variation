import sys
sys.path.insert(0, '../data')

import pybnb
import numpy as np
import math

from pvar_tools import *
from brute_force_warped_pvar import augment_path

class BnBWarping(pybnb.Problem):
    
    """ The solver in pybnb keeps track of the best solution seen so far for you, 
        and will prune the search space by not calling the branch() method when it 
        encounters a node whose bound() is worse than the best objective() seen so far.
    """

    def __init__(self, x, y, p, depth, norm='l1', plot_2d=False):

        self.x = np.array(x)
        self.m = len(self.x)
        self.y = np.array(y)
        self.n = len(self.y)
        assert self.m > 0
        assert self.n > 0
        self.plot_2d = plot_2d

        self.p = p
        self.depth = depth
        self.norm = norm

#         self.values_memoization = {}
        self.path = [(0,0), (0,0)]
        self.evaluation = [math.inf] # keep track of the history of parent function evaluations

    def align(self, warp):
        """align x and y according to the warping path"""
        x_reparam = np.array([self.x[k] for k in [j[0] for j in warp]])
        y_reparam = np.array([self.y[k] for k in [j[1] for j in warp]])
        return x_reparam, y_reparam

    def distance(self, warp):
        """computes warped p-variation along one path with dynamic programming algo"""
        x_reparam, y_reparam = self.align(warp)
        pvar, partition = p_variation_distance(x_reparam, y_reparam, p=self.p, 
                                               depth=self.depth, norm=self.norm)
        return pvar, partition
    
    #def random_exploratory_path(self, path):
    #    i,j = path[-1]
    #    if (i==self.m-1) and (j<self.n-1):
    #        new_path = list(path) + [(i,j+1)]
    #        return self.random_exploratory_path(new_path)
    #    elif (i<self.m-1) and (j==self.n-1):
    #        new_path = list(path) + [(i+1,j)]
    #        return self.random_exploratory_path(new_path)
    #    elif (i<self.m-1) and (j<self.n-1):
    #        direction = np.random.randint(3)
    #        if direction == 0:
    #            new_path = list(path) + [(i,j+1)]
    #        elif direction == 1:
    #            new_path = list(path) + [(i+1,j)]
    #        else:
    #            new_path = list(path) + [(i+1,j+1)]
    #        return self.random_exploratory_path(new_path)
    #    else:
    #        return path
        
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

        # TODO: do we need to keep track of values calculated along each path?
        assert len(self.evaluation) in (len(self.path), len(self.path)-1)
        if len(self.evaluation) == len(self.path)-1:
            self.evaluation.append(val)

        return val

    def bound(self):
        """ This function is evaluate at a partial path and needs to be a lower bound on any complete 
            path originating from it, so it can decide if the search needs to continue 
            along a partial path based on the best known objective.
        """
        
        #exploratory_path = self.random_exploratory_path(self.path)
        #print(exploratory_path)
        #return self.distance(exploratory_path)
        
#         return self.unbounded_objective()
        b, _ = self.distance(self.path)
        return b

    def save_state(self, node):
        node.state = (list(self.path), list(self.evaluation))

    def load_state(self, node):
        (self.path, self.evaluation) = node.state

    def branch(self):
        
        i,j = self.path[-1]
        
        if (i==self.m-1) and (j<self.n-1):
            child = pybnb.Node()
            child.state = (self.path + [(i,j+1)], list(self.evaluation))
            yield child
        
        elif (i<self.m-1) and (j==self.n-1):
            child = pybnb.Node()
            child.state = (self.path + [(i+1,j)], list(self.evaluation))
            yield child
        
        elif (i<self.m-1) and (j<self.n-1):
            child = pybnb.Node()
            child.state = (self.path + [(i+1,j)], list(self.evaluation))
            yield child
        
            child = pybnb.Node()
            child.state = (self.path + [(i,j+1)], list(self.evaluation))
            yield child
        
            child = pybnb.Node()
            child.state = (self.path + [(i+1,j+1)], list(self.evaluation))
            yield child
            
#     def notify_new_best_node(self, node, current=True):
#         print('we found a new best', node)

    def plot_alignment(self):
        if self.plot_2d:
            plt.plot(self.x.T[0], self.x.T[1], 'bo-' ,label = 'x')
            plt.plot(self.y.T[0], self.y.T[1], 'g^-', label = 'y')
        else:
            plt.plot(self.x, 'bo-' ,label = 'x')
            plt.plot(self.y, 'g^-', label = 'y')
        plt.title('Alignment')
        plt.legend()
        for map_x, map_y in self.path:
            if self.plot_2d:
                plt.plot([self.x[map_x][0], self.y[map_y][0]], [self.x[map_x][1], self.y[map_y][1]], 'r')
            else:
                plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')