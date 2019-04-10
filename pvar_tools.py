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

def sig_norm(sig, norm):
	"""calculate norm difference between two signatures"""
	assert norm in ['l1', 'l2']
	if norm == 'l1':
		return np.sum([np.abs(x) for x in sig])
	elif norm == 'l2':
		return np.sum([x*x for x in sig])

def p_variation_path(p, path, depth, norm='l1'):
    """return signature p-variation and optimal partition points of a path up to level depth using the given norm"""
    length = path.shape[0]
    dist = lambda a,b: sig_norm(signature(path[a:b+1,:],depth)[1:])
    pv = p_var_backbone(length, p, dist)
    return pv.value, pv.points

def pairwise_sig_norm(path1, path2, depth, a, b, norm):
	"""compute ||S(path1)_ab - S(path2)_ab||"""
	return sig_norm(signature(path1[a:b+1,:], depth) - signature(path2[a:b+1,:], depth), norm)

def p_variation_distance(path1, path2, p, depth, norm='l1'):
    """path1 and path2 must be numpy arrays of equal lenght.
	return the signature p-variation distance and points of the optimal partition""" 
    assert norm in ['l1', 'l2']
    assert path1.shape[0] == path2.shape[0]
    assert path1.shape[1] == path2.shape[1]
    length = path1.shape[0]
    dist = lambda a,b: pairwise_sig_norm(path1, path2, depth, a, b, norm)
    return p_var_backbone_ref(length, p, dist)

def p_variation_distance_alexey_optim(path1, path2, p, depth, norm='l1'):
    """path1 and path2 must be numpy arrays of equal lenght. Alexey's algo.
	return the signature p-variation distance and points of the optimal partition""" 
    assert norm in ['l1', 'l2']
    assert path1.shape[0] == path2.shape[0]
    assert path1.shape[1] == path2.shape[1]
    length = path1.shape[0]
    dist = lambda a,b: pairwise_sig_norm(path1, path2, depth, a, b, norm)
    pv = p_var_backbone(length, p, dist)
    return pv.value, pv.points

#def p_variation_distance_warwick(path1, path2, p, depth, norm='l1'):
#    """path1 and path2 must be numpy arrays of equal lenght. Warwick guy's algo.
#       return the signature p-variation distance and points of the optimal partition""" 
#    assert norm in ['l1', 'l2']
#    assert path1.shape[0] == path2.shape[0]
#    assert path1.shape[1] == path2.shape[1]
#    dist = lambda a,b: pairwise_sig_norm(path1, path2, depth, a, b, norm)
#    length = path1.shape[0]
#    optimal_partition = [0, 1]
#    p_variation = [dist(0,1)]
#    for i in range(1, length):
#        a = []
#        for j in range(i):
#            a.append(p_variation[j] + dist(j,i+1)**p)
#        t_m, a_m = max(enumerate(a), key=lambda x: x[1])
#        p_variation.append(a_m)
#        optimal_partition = optimal_partition[:t_m] + [i]
#    return pow(p_variation[-1],1./p), optimal_partition