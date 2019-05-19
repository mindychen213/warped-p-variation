import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns

class Warp:
    """ Inputs: x, y (lists)
                d (point-wise distance function)
                f_d (R x R x R -----> R)
        
        Return: Global_Warping_Distance (cost) (and many other functionalities)
        
        Axioms on f_d:
        1) Global_Warp_Distance(x[0...T], y[0...T]) = f_d(Global_Warp_Distance(x[0...T-1], y[0...T-1]), x[T], y[T])
        2) Symmetry, i.e. invariant to coordinate swapping
    """
    
    def __init__(self, x, y, d, f_d, final_operator):
    
        self.x = x
        self.y = y
        self.x_nsamples = len(x)
        self.y_nsamples = len(y)

        self.final_operator = final_operator
        self.d = d
        self.f_d = f_d
        
        self.D = self._accumulated_cost()
        
        self.warping_path = [[len(x)-1, len(y)-1]]
        self.cost = 0.
        self._back_track()
    
    def pairwise_distances(self):
        distances = np.zeros((self.y_nsamples, self.x_nsamples))
        for i in range(self.y_nsamples):
            for j in range(self.x_nsamples):
                distances[i,j] = self.d(self.x[j], self.y[i])
        return distances
        
    def distance_cost_plot(self, distances):
        im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()
        plt.title('distances')
    
    def _accumulated_cost(self):
        DD = np.zeros((self.y_nsamples, self.x_nsamples))
        DD[0,0] = self.d(self.x[0], self.y[0])
        
        # If we were to move along the first row, i.e. from (0,0) in the right direction only, one step at a time
        for j in range(1, self.x_nsamples):
            DD[0,j] = self.f_d(DD[0, j-1], self.x[j], self.y[0]) 

        # If we were to move along the first column, i.e. from (0,0) in the upwards direction only, one step at a time
        for i in range(1, self.y_nsamples):
            DD[i, 0] = self.f_d(DD[i-1, 0], self.x[0], self.y[i])
            
        # Accumulated Cost: D(i,j) = min{f_d(D(i−1,j−1), x[i], y[j]), f_d(D(i−1,j), x[i], y[j]), f_d(D(i,j−1), x[i], y[j])}
        for i in range(1, self.y_nsamples):
            for j in range(1, self.x_nsamples):
                DD[i, j] = min(self.f_d(DD[i-1, j-1], self.x[j], self.y[i]),
                               self.f_d(DD[i-1, j], self.x[j], self.y[i]),
                               self.f_d(DD[i, j-1], self.x[j], self.y[i]))
    
        return DD
    
    def _back_track(self):    
        i = self.y_nsamples - 1
        j = self.x_nsamples - 1
        
        while i>0 and j>0:
            if i == 0:
                j = j - 1
            elif j == 0:
                i = i - 1
            else:
                if self.D[i-1, j] == min(self.D[i-1, j-1], self.D[i-1, j], self.D[i, j-1]):
                    i = i - 1
                elif self.D[i, j-1] == min(self.D[i-1, j-1], self.D[i-1, j], self.D[i, j-1]):
                    j = j - 1
                else:
                    i = i - 1
                    j= j - 1
            self.warping_path.append([j, i])
        self.warping_path.append([0,0])
        
        for [p, q] in self.warping_path:
            self.cost += self.d(self.x[p], self.y[q])
        self.cost = self.final_operator(self.cost)

    def plot_lattice(self):
        self.distance_cost_plot(self.D)
        plt.plot([e[0] for e in self.warping_path], [e[1] for e in self.warping_path])
        plt.title('Warping')
    
    def plot_alignment(self):
        #plt.plot(self.x.T[0], self.x.T[1], 'bo-' ,label = 'x')
        #plt.plot(self.y.T[0], self.y.T[1], 'g^-', label = 'y')
        plt.plot(self.x, 'bo-' ,label = 'x')
        plt.plot(self.y, 'g^-', label = 'y')
        plt.title('Alignment')
        plt.legend()
        for [map_x, map_y] in self.warping_path:
            #plt.plot([self.x[map_x][0], self.y[map_y][0]], [self.x[map_x][1], self.y[map_y][1]], 'r')
            plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')