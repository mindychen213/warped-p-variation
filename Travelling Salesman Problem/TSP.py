from Node import Node
import math
import time
import copy

class TSP():

    def __init__(self, size, costs, bestTour=math.inf):
        self.size = size
        self.costs = costs
        self.bestTour = bestTour
        self.Node = None
        self.bestNodeTime = 0
        self.num_createdNodes = 0
        self.num_prunedNodes = 0
        self.sortedEdges = self.sort_edges()
        self.allSortedEdges = self.sort_allEdges()

    def findSolution(self):
        root  = self.create_root()
        self.num_createdNodes += 1
        T1 = time.perf_counter()
        self.BranchAndBound(root)
        T2 = time.perf_counter()
        print('---------------')
        print('The shortest tour is: ', self.bestNode)
        print('It has a length of: ', self.bestTour, ' km')
        print('Found in ', T2-T1, ' sec')
        print('Best tour was found after: ', self.bestNodeTime, ' sec')
        print('Number of nodes created: ', self.num_createdNodes)
        print('Number of nodes pruned: ', self.num_prunedNodes)

    def sort_edges(self):
        """sorts edges of the distance matrix per row and returns 
           matrix where each row i contains the numbers 0<=k<=self.size-1
           in the order of increasing costs of the edges (i,k)
        """
        result = []
        for i in range(self.size):
            result.append([x for (y,x) in sorted(zip(self.costs[i],
                                                     list(range(self.size))
                                                     )
                                                 )
                           ]
                          )
        return result

    def sort_allEdges(self):
        """sorts all edges of distance matrix and returns list of pairs
            (i,j) in order of increasing costs
        """
        edges = []
        lengths = []
        for i in range(self.size):
            for j in range(i+1, self.size):
                edges.append([i,j])
                lengths.append(c[i][j])
                # lengths.append(self.costs[i][j])
        return [z for (l,z) in sorted(zip(lengths, edges))]

    def create_root(self):
        no_constraints = []
        for i in range(self.size):
            row_i = []
            for j in range(self.size):
                if i!=j:
                    row_i.append(2)
                else:
                    row_i.append(0)
            no_constraints.append(row_i)
        root = Node(self.size, 
                    self.costs, 
                    self.sortedEdges,
                    self.allSortedEdges,
                    no_constraints)
        return root

    