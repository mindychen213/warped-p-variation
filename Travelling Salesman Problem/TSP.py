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

