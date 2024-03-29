from pvar_backend import *
import numpy as np
from esig import tosig
import iisignature
from numba import njit
from functools import lru_cache

def signature(path, depth):
	"""return signature of a path up to level depth"""
	assert isinstance(depth, int), 'Argument of wrong type'
	assert depth > 1, 'depth must be > 1'
	width = path.shape[1]
	length = path.shape[0]
	if length <= 1:
		return np.array([0.]*(tosig.sigdim(width, depth)-1))
	return iisignature.sig(path, depth)

@njit
def numba_l1(x, y):
    res = 0.
    for x_i, y_i in zip(x, y):
        res = res + abs(x_i - y_i)
    return res

@njit
def numba_l2(x, y):
    res = 0.
    for x_i, y_i in zip(x, y):
        res += (x_i - y_i)*(x_i - y_i)
    return res**(0.5)

def sig_norm(sig_x, sig_y, norm='l1'):
	"""calculate norm difference between two signatures"""
	assert norm in ['l1', 'l2']
	if norm == 'l1':
		return numba_l1(sig_x, sig_y)
	elif norm == 'l2':
		return numba_l2(sig_x, sig_y)

def p_variation_path(p, path, depth, norm='l1', optim_partition=True, pvar_advanced=False):
    """return signature p-variation and optimal partition points of a path up to level
       depth using the given norm and Dynamic Programming algorithm"""
    length = path.shape[0]
    dist = lambda a,b: sig_norm(signature(path[:b+1,:],depth), signature(path[:a+1,:],depth), norm)
    if pvar_advanced:
        return p_var_backbone(length, p, dist, optim_partition)
    return p_var_backbone_ref(length, p, dist, optim_partition)

def pairwise_sig_norm(path1, path2, depth, a, b, norm='l1'):
	"""compute ||S(path1)_ab - S(path2)_ab||"""
	return sig_norm(signature(path1[a:b+1,:], depth), signature(path2[a:b+1,:], depth), norm)

def p_variation_distance(path1, path2, p, depth, norm='l1', optim_partition=False, pvar_advanced=False):
    """path1 and path2 must be numpy arrays of equal lenght.
	return the signature p-variation distance and points of the optimal partition""" 
    assert norm in ['l1', 'l2']
    assert len(path1) == len(path2)
    length = len(path1)
    dist = lambda a,b: pairwise_sig_norm(path1, path2, depth, a, b, norm)
    if pvar_advanced:
        return p_var_backbone(length, p, dist, optim_partition)
    return p_var_backbone_ref(length, p, dist, optim_partition)

#@lru_cache(maxsize=None)
#def sig_pvar(x_reparam, y_reparam, p, norm, depth):
#    m = len(x_reparam)
#    n = len(y_reparam)
#    assert m == n 
#    if m == 1:
#        return 0.
#    final = 0.
#    for i in range(1,m):
#        sig_x = signature(x_reparam[:i+1], depth)
#        sig_y = signature(y_reparam[:i+1], depth)
#        sig_n = sig_norm(sig_x, sig_y, norm)**p
#        d = sig_pvar(x_reparam[i:], y_reparam[i:], p, norm, depth)**p
#        dpvar = sig_n + d
#        if final < dpvar:
#            final = dpvar
#    return final**(1./p)