import numpy as np
from pvar_backend import *
from paths_tools import *
import unittest

class Tests(unittest.TestCase):

    def _p_var_points_check(self, p_var_ret, p, path_dist):
        # Check the output of p_var_backbone: Returns abs value of the error.
        # Whether the p-variation p_var_ret.value is indeed reached on the sequence p_var_ret.points.
        v = 0.0
        for k in range(1, len(p_var_ret.points)):
            v += pow(path_dist(p_var_ret.points[k-1], p_var_ret.points[k]), p)
        return abs(pow(v,1./p) - p_var_ret.value)

    def test_random_path(self, epsilon=1e-1):
        # Example: Brownian motion made of iid -1/+1 increments
        path = np.random.rand(100, 2)
        dist = lambda a, b: np.sum([np.abs(x) for x in (path[b,:] - path[a,:])])
        for p in [1.0, math.sqrt(2), 2.0, math.exp(1)]:          
            pv = p_var_backbone(len(path), p, dist)
            pv_ref = p_var_backbone_ref(len(path), p, dist)
            # check the two methods agree
            pv_err_pval = abs(pv.value - pv_ref) 
            self.assertGreater(epsilon, pv_err_pval)
            # check that the partition is optimal
            pv_err_part = self._p_var_points_check(pv, p, dist)
            self.assertGreater(epsilon, pv_err_part)

    def test_bm(self, epsilon=1e-1):
        # Example: Brownian motion made of iid -1/+1 increments
        n = 2500
        path = [0.0] * (n + 1)
        sigma = 1. / math.sqrt(n)
        for k in range(1, n + 1):
            path[k] = path[k-1] + random.choice([-1, 1]) * sigma
        dist = lambda a, b: abs(path[b] - path[a])
        for p in [1.0, math.sqrt(2), 2.0, math.exp(1)]:          
            pv = p_var_backbone(len(path), p, dist)
            pv_ref = p_var_backbone_ref(len(path), p, dist)
            # check the two methods agree
            pv_err_pval = abs(pv.value - pv_ref) 
            self.assertGreater(epsilon, pv_err_pval)
            # check that the partition is optimal
            pv_err_part = self._p_var_points_check(pv, p, dist)
            self.assertGreater(epsilon, pv_err_part)

    def test_warped_pvar(self, epsilon=2e-1):
        # Test global warping distance consistency between Alexey's algo and standard DP
        x = np.random.rand(4, 2)
        y = np.random.rand(3, 2)
        LP = LatticePaths(x, y, p=2., depth=2, norm='l1', brute_force=True)
        p1, _ = LP.optim_warped_pvar
        p2 = LP.warped_pvar
        self.assertGreater(epsilon, np.abs(p1-p2))

if __name__=="__main__":
    unittest.main()