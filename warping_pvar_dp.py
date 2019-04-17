import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pvar_tools import sig_norm, signature, p_variation_distance

# d_W(X_p,Y_p) = d_W((X_1,...,X_T), (Y_1, ..., Y_T))
#
#              = F(d_W((X_1,...,X_T-1), (Y_1, ..., Y_T-1)), X_T, Y_T)
#
#              = max(
#                       ||S(X)_0,T - S(Y)_0,T||,
#
#                       (d_W((X_1,...,X_2), (Y_1, ..., Y_2))**p + ||S(X)_2,T - S(Y)_2,T||**p)^(1/p), 
#
#                       (d_W((X_1,...,X_3), (Y_1, ..., Y_3))**p + ||S(X)_3,T - S(Y)_3,T||**p)^(1/p),
#                       
#                       ...
#                       
#                       (d_W((X_1,...,X_T-2), (Y_1, ..., Y_T-2))**p + ||S(X)_T-2,T - S(Y)_T-2,T||**p)^(1/p),
#
#                       (d_W((X_1,...,X_T-1), (Y_1, ..., Y_T-1))**p + ||S(X)_T-1,T - S(Y)_T-1,T||**p)^(1/p)
#                   
#                   )
#              
#              = F() 

class WarpingPvar:
    """ Inputs: x, y (lists)
                d (point-wise distance function)
                f_d (R x R x R -----> R)
        
        Return: .......
    """
    
    def __init__(self, x, y, p=1.5, depth=2, norm='l1'):
    
        self.x = x
        self.y = y
        self.x_nsamples = len(x)
        self.y_nsamples = len(y)
        self.p = p
        self.depth = depth
        self.norm = norm

        self.pvariations_memoizer = defaultdict(tuple)
        self.best_warp = defaultdict(tuple)
        
        self._pvar_dynamic_programming()
        self.warped_pvar_dist = self.warping_dynamic_programming()
             
    def dynamic_step(self, a, b):
        return pow(a**self.p + b**self.p, 1./self.p)

    def align(self, warp):
        """align x and y according to the warping path"""
        x_reparam = np.array([self.x[k] for k in [s[0] for s in warp]])
        y_reparam = np.array([self.y[k] for k in [s[1] for s in warp]])
        return x_reparam, y_reparam

    def concat(self, warp, j, i):
        return warp + (j, i)

    def back_track(self, warp, i, j):
        store_pvars = []
        for jj, ii in warp[::-1]:
            store_pvars.append(self.dynamic_step(self.pvariations_memoizer[(ii,jj)],
                               sig_norm(signature(self.x[jj:j+1,:], self.depth) - signature(self.y[ii:i+1,:], self.depth)))
                               )
        return max(store_pvars)

    def _pvar_dynamic_programming(self):
        # initialization
        self.pvariations_memoizer[(0,0)] = sig_norm(signature(self.x, self.depth) - signature(self.y, self.depth), self.norm)
        
        # If we were to move along the first row, i.e. from (0,0) in the right direction only, one step at a time
        for j in range(1, self.x_nsamples):
                        
            self.pvariations_memoizer[(0,j)] = self.back_track(self.best_warp[(0,j-1)], 0, j)
            #xa = self.dynamic_step(self.pvariations_memoizer[0, j-1], sig_norm(signature(self.x[j-1:j+1,:], self.depth)))
            #pathx1, pathx2 = self.align(self.concat(self.best_warp[j-1, 0], j, 0))
            #xb = p_variation_distance(pathx1, pathx2, self.p, self.depth, self.norm)
            #self.pvariations_memoizer[0, j] = max(xa, xb)

        # If we were to move along the first column, i.e. from (0,0) in the upwards direction only, one step at a time
        for i in range(1, self.y_nsamples):
            
            self.pvariations_memoizer[(i,0)] = self.back_track(self.best_warp[(i-1,0)], i, 0)
            #ya = self.dynamic_step(self.pvariations_memoizer[i-1, 0], sig_norm(signature(self.y[i-1:i+1,:], self.depth)))
            #pathy1, pathy2 = self.align(self.concat(self.best_warp[0, i-1], 0, i))
            #yb = p_variation_distance(pathy1, pathy2, self.p, self.depth, self.norm)
            #self.pvariations_memoizer[i, 0] = max(ya, yb)
            
        # diagonal step
        for i in range(1, self.y_nsamples):
            for j in range(1, self.x_nsamples):

                #a1 = self.dynamic_step(self.pvariations_memoizer[i-1, j-1], 
                #                       sig_norm(signature(self.x[j-1:j+1,:], self.depth) - signature(self.y[i-1:i+1,:], self.depth)))
                #patha1, pathb1 = self.align(self.concat(self.best_warp[j-1,i-1], j, i))
                #b1 = p_variation_distance(patha1, pathb1, self.p, self.depth, self.norm)

                #a2 = self.dynamic_step(self.pvariations_memoizer[i-1, j], 
                #                       sig_norm(signature(self.x[j:j+1,:], self.depth) - signature(self.y[i-1:i+1,:], self.depth)))
                #patha2, pathb2 = self.align(self.concat(self.best_warp[j,i-1], j, i))
                #b2 = p_variation_distance(patha2, pathb2, self.p, self.depth, self.norm)

                #a3 = self.dynamic_step(self.pvariations_memoizer[i, j-1], 
                #                      sig_norm(signature(self.x[j-1:j+1,:], self.depth) - signature(self.y[i:i+1,:], self.depth)))
                #patha3, pathb3 = self.align(self.concat(self.best_warp[j-1,i], j, i))
                #b3 = p_variation_distance(patha3, pathb3, self.p, self.depth, self.norm)

              
                self.pvariations_memoizer[i, j] = min(self.back_track(self.best_warp[(i, j-1)], i, j), 
                                                      self.back_track(self.best_warp[(i-1, j)], i, j),
                                                      self.back_track(self.best_warp[(i-1, j-1)], i, j))

    
    def warping_dynamic_programming(self):    
        i = self.y_nsamples - 1
        j = self.x_nsamples - 1
        
        while i>0 and j>0:
            if i == 0:
                j = j - 1
            elif j == 0:
                i = i - 1
            else:
                if self.pvariations_memoizer[i-1, j] == min(self.pvariations_memoizer[i-1, j-1], 
                                                            self.pvariations_memoizer[i-1, j], 
                                                            self.pvariations_memoizer[i, j-1]):
                    i = i - 1

                elif self.pvariations_memoizer[i, j-1] == min(self.pvariations_memoizer[i-1, j-1], 
                                                              self.pvariations_memoizer[i-1, j], 
                                                              self.pvariations_memoizer[i, j-1]):
                    j = j - 1

                else:
                    i = i - 1
                    j= j - 1

            self.best_warp.append([j, i])
        self.best_warp.append([0,0])
        
        self.best_warp.reverse()
        final_pathx, final_pathy = self.align(self.best_warp)

        return p_variation_distance(final_pathx, final_pathy, self.p, self.depth, self.norm)
    
    def plot_lattice(self):
        plt.plot([e[0] for e in self.best_warp], [e[1] for e in self.best_warp])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.title('Best Warp')
    
    def plot_alignment(self):
        plt.plot(self.x, 'bo-' ,label = 'x')
        plt.plot(self.y, 'g^-', label = 'y')
        plt.title('Alignment')
        plt.legend()
        for [map_x, map_y] in self.best_warp:
            plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')