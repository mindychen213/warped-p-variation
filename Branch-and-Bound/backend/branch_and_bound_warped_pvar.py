import pybnb
import numpy as np

class Lattice(pybnb.Problem):
    
    """ The solver in pybnb keeps track of the best solution seen so far for you, 
        and will prune the search space by not calling the branch() method when it 
        encounters a node whose bound() is worse than the best objective() seen so far.
    """

    def __init__(self, x, y):
        self.x = tuple(x)
        self.m = len(self.x)
        self.y = tuple(y)
        self.n = len(self.y)
        assert self.m > 0
        assert self.n > 0

#         self.values_memoization = {}
        self.path = [(0,0)]
        self.evaluation = [] # keep track of the history of parent function evaluations

    def align(self, path):
        """align x and y according to the path"""
        x_reparam = [self.x[k] for k in [i for i,j in path]]
        y_reparam = [self.y[k] for k in [j for i,j in path]]
        return x_reparam, y_reparam
    
    def distance(self, path):
        x, y = self.align(path)
        return np.sqrt(np.sum([(xx-yy)**2 for xx,yy in zip(x,y)]))
    
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
            val = self.distance(self.path)
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
        return self.distance(self.path)

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