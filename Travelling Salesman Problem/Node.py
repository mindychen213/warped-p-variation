import math
import copy
import numpy as np

class Node():
    def __init__(self, size, costs, sortedEdges, allSortedEdges, parent_constr, extra_constr=None):
        self.size = size # Number of cities
        self.costs = costs # Distance matrix
        self.sortedEdges = sortedEdges
        self.allSortedEdges = allSortedEdges
        self.extra_constr = extra_constr
        self.constraints = self.determine_constr(parent_constr)
        self.lowerBound = self.computeLowerBound()

    def computeLowerBound(self):
        lb = 0
        for i in range(self.size):
            lower = 0
            t = 0
            for j in range(self.size):
                if self.constraints[i][j] == 1:
                    lower += self.costs[i][j]
                    t += 1
            tt = 0
            while t < 2:
                shortest = self.sortedEdges[i][tt]
                if self.constraints[i][shortest] == 2:
                    lower += self.costs[i][shortest]
                    t += 1
                tt += 1
                if tt >= self.size:
                    lower = math.inf
                    break
            lb += lower
        return lb

    def determine_constr(self, parent_constr):
        """This function determines the constraints of a node based on the constraints
           of the parent and the extra new constraint for this node
        """
        constraints = copy.deepcopy(parent_constr)
        if self.extra_constr == None:
            return constraints
        fr = self.extra_constr[0]
        to = self.extra_constr[1]
        constraints[fr][to] = self.extra_constr[2]
        constraints[to][fr] = self.extra_constr[2]
        for i in range(2):
            constraints = self.removeEdges(constraints)
            constraints = self.addEdges(constraints)
        return constraints

    def removeEdges(self, constraints):
        """This function excludes edges when:
           1) two other edges adjacent to the same vertex are already included
           2) including the edge would create a sub-tour
        """
        for i in range(self.size):
            t = 0
            for j in range(self.size):
                if (i!=j) and (constraints[i][j]==1):
                    t+=1
            if t >= 2:
                for j in range(self.size):
                    if constraints[i][j] == 2:
                        constraints[i][j] = 0
                        constraints[j][i] = 0
        for i in range(self.size):
            for j in range(self.size):
                t = 1
                prev = i
                fr = j 
                cycle = False
                nextOne = self.next_one(prev, fr, constraints)
                while (nextOne[0]):
                    t += 1
                    next = nextOne[1]
                    if next == i:
                        cycle = True
                        break
                    if t > self.size:
                        break
                    prev = fr
                    fr = next
                    nextOne = self.next_one(prev, fr, constraints)
                if (cycle) and (t<self.size) and (constraints[i][j]==2):
                    constraints[i][j] = 0
                    constraints[j][i] = 0
        return constraints

    def addEdges(self, constraints):
        """This function checks if all but two edges adjacent to a vertex are excluded
           If so, these edges are included
        """
        for i in range(self.size):
            t = 0
            for j in range(self.size):
                if constraints[i][j] == 0:
                    t+=1
            if t ==  self.size - 2:
                for j in range(self.size):
                    if constraints[i][j] == 2:
                        constraints[i][j] = 1
                        constraints[j][i] = 1
        return constraints

    def next_one(self, prev, fr, constraints):
        """This function determines whether there exists an included edge that starts
           in fr and does end in prev. If so, it also returns the endpoint of this edge
        """ 
        #try:
        for j in range(self.size):
            if (constraints[fr][j] == 1) and (j != prev):
                return [True, j]
        return [False]
        #except TypeError:

    def isTour(self):
        """This function determines if a node represents a full tour by checking
           whether from every vertex there are exactly 2 included edges and all the
           other edges are excluded
        """
        for i in range(self.size):
            num_zero = 0
            num_one = 0
            for j in range(self.size):
                if self.constraints[i][j] == 0:
                    num_zero += 1
                elif self.constraints[i][j] == 1:
                    num_one += 1
            if (num_zero != self.size-2) or (num_one != 2):
                return False
        return True

    def contains_subtour(self):
        """checks if a node contains a subtour"""
        for i in range(self.size):
            next = self.next_one(i,i,self.constraints)
            if next[0]:
                prev=i
                fr=next[1]
                t =1
                next = self.next_one(prev,fr,self.constraints)
            while next[0]:
                t+=1
                prev=fr
                fr=next[1]
                if (fr==i) and (t<self.size):
                    return True
                next = self.next_one(prev,fr,self.constraints)
                if t==self.size:
                    return False
        return False

    def tourLength(self):
        """Assumes the node represents a tour and returns the length of this tour"""
        length = 0
        fr = 0
        to = self.next_one(fr, fr, self.constraints)
        for i in range(self.size):
            length += self.costs[fr][to]
            temp=fr
            fr=to 
            to = self.next_one(temp, to, self.constraints)[1]
        return length

    def next_constraint(self):
        """This function determines the next constraint according to the 
            branching strategy lexicographic order
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.constraints[i][j] == 2:
                    return [i,j]

    def __str__(self):
        """if a node represents a tour, this function returns a string
           with the order of the vertices on the tour
        """
        if self.isTour():
            result = '0'
            fr = 0
            to = None
            for j in range(self.size):
                if self.constraints[fr][j] == 1:
                    to = j 
                    result += '-' + str(j)
                    break
            for i in range(self.size-1):
                for j in range(self.size):
                    if (self.constraints[to][j]==1) and (j!=fr):
                        result += '-' + str(j)
                        fr = to 
                        to  = j
                        break
            return results
        else:
            print('This node is not a tour')

