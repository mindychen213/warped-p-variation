from pvar_backend import *
import numpy as np
from esig import tosig
import iisignature

def signature(path, depth):
	"""return signature of a path up to level depth"""
	assert isinstance(depth, int), 'Argument of wrong type'
	assert depth > 1, 'depth must be > 1'
	width = path.shape[1]
	length = path.shape[0]
	if length <= 1:
		return np.array([0.]*(tosig.sigdim(width, depth)-1))
	return iisignature.sig(path, depth)

def sig_norm(sig, norm='l1'):
	"""calculate norm difference between two signatures"""
	assert norm in ['l1', 'l2']
	if norm == 'l1':
		return sum([abs(x) for x in sig])
	elif norm == 'l2':
		return sum([x*x for x in sig])

def p_variation_path(p, path, depth, norm='l1', optim_partition=True):
    """return signature p-variation and optimal partition points of a path up to level
       depth using the given norm and Dynamic Programming algorithm"""
    length = path.shape[0]
    dist = lambda a,b: sig_norm(signature(path[:b+1,:],depth) - signature(path[:a+1,:],depth), norm)
    return p_var_backbone_ref(length, p, dist, optim_partition)

#def p_variation_path_optim(p, path, depth, norm='l1'):
#    """return signature p-variation and optimal partition points of a path up to level 
#       depth using the given norm and Alexey's optimised algorithm"""
#    length = path.shape[0]
#    dist = lambda a,b: sig_norm(signature(path[:b+1,:],depth) - signature(path[:a+1,:],depth), norm)
#    return p_var_backbone(length, p, dist)

def pairwise_sig_norm(path1, path2, depth, a, b, norm='l1'):
	"""compute ||S(path1)_ab - S(path2)_ab||"""
	return sig_norm(signature(path1[a:b+1,:], depth) - signature(path2[a:b+1,:], depth), norm)

def p_variation_distance(path1, path2, p, depth, norm='l1', optim_partition=False):
    """path1 and path2 must be numpy arrays of equal lenght.
	return the signature p-variation distance and points of the optimal partition""" 
    assert norm in ['l1', 'l2']
    assert len(path1) == len(path2)
    length = len(path1)
    dist = lambda a,b: pairwise_sig_norm(path1, path2, depth, a, b, norm)
    return p_var_backbone_ref(length, p, dist, optim_partition)

#def p_variation_distance_optim(path1, path2, p, depth, norm='l1'):
#    """path1 and path2 must be numpy arrays of equal lenght. Alexey's algo.
#	return the signature p-variation distance and points of the optimal partition""" 
#    assert norm in ['l1', 'l2']
#    assert len(path1) == len(path2)
#    length = len(path1)
#    dist = lambda a,b: pairwise_sig_norm(path1, path2, depth, a, b, norm)
#    return p_var_backbone(length, p, dist)