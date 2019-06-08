import sys
sys.path.insert(0, '../data')

import pybnb
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

import pvar_backend
import pvar_tools 

class BnBWarping(pybnb.Problem):
    
    """ The solver in pybnb keeps track of the best solution seen so far for you, 
        and will prune the search space by not calling the branch() method when it 
        encounters a node whose bound() is worse than the best objective() seen so far.
    """

    def __init__(self, x, y, p, depth, norm='l1', root_node=(0, 0), boundary_condition=1,
                 plot_2d=False, use_dp=True, with_sig_memoization=True, pvar_advanced=False, pth_root=False, 
                 sig_memoizer_x={}, sig_memoizer_y={}, norm_pairs_memoizer={}, bound_memoizer={}):

        self.x = np.array(x)
        self.y = np.array(y)
 
        self.m = len(self.x)
        self.n = len(self.y)
        
        assert self.m > 0
        assert self.n > 0

        self.plot_2d = plot_2d # flag to plot paths in 1-d or 2-d case
        self.use_dp = use_dp # flag for enable usage of dynamic programming logic
        self.with_sig_memoization = with_sig_memoization

        # dynamic programming is only allowed to kick in if we are at least boundary_conditions nodes away from root.
        self.boundary_condition = boundary_condition

        self.p = p # p for p-variation
        self.depth = depth # signature depth
        self.norm = norm # l1 or l2 norm

        self.pvar_advanced = pvar_advanced # use Alexey's algorithm for p-variation
        self.pth_root = pth_root # take pth root of p-variation

        self.path = [(0,0), (0,0)] # lattice path
        self.best_node_value = math.inf # keep track of best bound

        self.i0, self.j0 = root_node # tuple of indeces of root node
        
        # all these variables are global, hence shared by all recursive calls to the class
        self.sig_memoizer_x = sig_memoizer_x
        self.sig_memoizer_y = sig_memoizer_y 
        self.norm_pairs_memoizer = norm_pairs_memoizer
        self.bound_memoizer = bound_memoizer 
 

    def align(self, warp):
        """align x and y according to the warping path"""
        x_reparam = np.array([self.x[k] for k in [j[0] for j in warp]])
        y_reparam = np.array([self.y[k] for k in [j[1] for j in warp]])
        return x_reparam, y_reparam

    def distance(self, warp, compute_optim_partition=False):
        """computes warped p-variation along one path with dynamic programming algo"""

        if self.with_sig_memoization:

            index_x_reparam = [j[0] for j in warp]
            index_y_reparam = [j[1] for j in warp]
            length = len(warp)

            def dist(a, b):

                i_0, i_N = index_x_reparam[a], index_x_reparam[b]+1
                j_0, j_N = index_y_reparam[a], index_y_reparam[b]+1

                if (i_0+self.i0, j_0+self.j0, i_N+self.i0, j_N+self.j0) in self.norm_pairs_memoizer:
                    return self.norm_pairs_memoizer[(i_0+self.i0, j_0+self.j0, i_N+self.i0, j_N+self.j0)]

                if (i_0+self.i0, i_N+self.i0) in self.sig_memoizer_x:
                    sig_x = self.sig_memoizer_x[(i_0+self.i0, i_N+self.i0)]
                else:
                    sig_x = pvar_tools.signature(self.x[i_0:i_N], self.depth)
                    self.sig_memoizer_x[(i_0+self.i0, i_N+self.i0)] = copy.deepcopy(sig_x)

                if (j_0+self.j0, j_N+self.j0) in self.sig_memoizer_y:
                    sig_y = self.sig_memoizer_y[(j_0+self.j0, j_N+self.j0)]
                else:
                    sig_y = pvar_tools.signature(self.y[j_0:j_N], self.depth)
                    self.sig_memoizer_y[j_0+self.j0, j_N+self.j0] = copy.deepcopy(sig_y)

                s_norm = pvar_tools.sig_norm(sig_x, sig_y, self.norm)
                self.norm_pairs_memoizer[(i_0+self.i0, j_0+self.j0, i_N+self.i0, j_N+self.j0)] = copy.deepcopy(s_norm)
                return s_norm
            
            if self.pvar_advanced:
                pvar, partition = pvar_backend.p_var_backbone(length, self.p, dist, compute_optim_partition, self.pth_root)
            pvar, partition = pvar_backend.p_var_backbone_ref(length, self.p, dist, compute_optim_partition, self.pth_root)

        else:
            x_reparam, y_reparam = self.align(warp)
            pvar, partition = pvar_tools.p_variation_distance(x_reparam, y_reparam, p=self.p, depth=self.depth, 
                                                              norm=self.norm, optim_partition=compute_optim_partition, 
                                                              pvar_advanced=self.pvar_advanced, pth_root=self.pth_root)
        
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
        """ This function is evaluated at a partial path and needs to be a lower bound on any complete 
            path originating from it, so it can decide if the search needs to continue 
            along a partial path based on the best known objective.
        """
        
        #return self.unbounded_objective()

        a = 0. # quantity to be added to soft bound to form tight bound (i.e. solution to sub-problem)
        b, _ = self.distance(self.path) # warped p-variation distance along path so far (soft bound)
        i,j = self.path[-1] # current position in lattice

        if self.use_dp: # use tight bound (dynamic programming within BnB) if necessary

            # if the soft bound is not enough then compute the tight bound
            if b < self.best_node_value: 

                # Boundary condition: use tight bound only if we are at least boundary_condition nodes away from root
                if (i<self.boundary_condition) or (j<self.boundary_condition):
                    a = 0.
                else:
                    # Bound memoization
                    if (i+self.i0, j+self.j0) in self.bound_memoizer:
                        a = self.bound_memoizer[(i+self.i0, j+self.j0)]
                    else:
                        sub_x = self.x[i:]
                        sub_y = self.y[j:]

                        sub_problem = BnBWarping(x=sub_x, y=sub_y, p=self.p, depth=self.depth, norm=self.norm, 
                                                 root_node=(i,j),
                                                 #root_node=(0,0),
                                                 boundary_condition=1, use_dp=self.use_dp, 
                                                 with_sig_memoization=self.with_sig_memoization, 
                                                 pvar_advanced=self.pvar_advanced, pth_root=self.pth_root, 
                                                 sig_memoizer_x=self.sig_memoizer_x, 
                                                 sig_memoizer_y=self.sig_memoizer_y, 
                                                 norm_pairs_memoizer=self.norm_pairs_memoizer, 
                                                 bound_memoizer=self.bound_memoizer)

                        results = pybnb.Solver().solve(sub_problem, log=None, queue_strategy='depth')

                        a = results.objective
                        self.bound_memoizer[(i+self.i0, j+self.j0)] = copy.deepcopy(a)

        if self.pth_root:
            return (b**self.p + a**self.p)**(1./self.p)
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