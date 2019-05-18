class Lattice(pybnb.Problem):

    def __init__(self, x, y):
        self.x = tuple(x)
        self.m = len(self.x)
        self.y = tuple(y)
        self.n = len(self.y)
        assert self.m > 0
        assert self.n > 0

        self.path = [(0,0)]
        self.evaluation = []
        # TODO
#         self.values_memoization = {}

    def distance(self, path):
        return np.sum([(self.x[i] - self.y[j])**2 for i,j in path])
    
    def random_exploratory_path(self, path):
        i,j = path[-1]
        if (i==self.m-1) and (j<self.n-1):
            new_path = list(path) + [(i,j+1)]
            return self.random_exploratory_path(new_path)
        elif (i<self.m-1) and (j==self.n-1):
            new_path = list(path) + [(i+1,j)]
            return self.random_exploratory_path(new_path)
        elif (i<self.m-1) and (j<self.n-1):
            direction = np.random.randint(3)
            if direction == 0:
                new_path = list(path) + [(i,j+1)]
            elif direction == 1:
                new_path = list(path) + [(i+1,j)]
            else:
                new_path = list(path) + [(i+1,j+1)]
            return self.random_exploratory_path(new_path)
        else:
            return path
        
    def sense(self):
        return pybnb.minimize

    def objective(self):
#         val = self.infeasible_objective()
        val = self.distance(self.path)
        assert len(self.evaluation) in (len(self.path), len(self.path)-1)
        if len(self.evaluation) == len(self.path)-1:
            self.evaluation.append(val)
        return val

    def bound(self):
        exploratory_path = self.random_exploratory_path(self.path)
        print(exploratory_path)
#         return self.unbounded_objective()
        return self.distance(exploratory_path)

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