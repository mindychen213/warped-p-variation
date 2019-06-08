import math
import collections
import random
import time
import copy
import numpy as np


def p_var_backbone_ref(path_size, p, path_dist, optim_partition=False, pth_root=True):
    # p-variation via Dynamic Programming

    if path_size == 0:
        return -math.inf
    elif path_size == 1:
        return 0

    cum_p_var = [0.] * path_size
    point_links = [0] * path_size
    for j in range(1, path_size):       
        for k in range(j):
            temp = path_dist(k, j)**p + cum_p_var[k]
            if cum_p_var[j] < temp:
                cum_p_var[j] = temp
                point_links[j] = k

    if optim_partition:
        points = []
        point_i = path_size-1
        while True:
            points.append(point_i)
            if point_i == 0:
                break
            point_i = point_links[point_i]
        points.reverse()
    else:
        points = []

    if pth_root:
        return cum_p_var[-1]**(1./p), points
    return cum_p_var[-1], points



def p_var_backbone(path_size, p, path_dist, optim_partition=False, pth_root=True):
    # Input:
    # * path_size >= 0 integer
    # * p >= 1 real
    # * path_dist: metric on the set {0,...,path_dist-1}.
    #   Namely, path_dist(a,b) needs to be defined and nonnegative
    #   for all integer 0 <= a,b < path_dist, be symmetric and
    #   satisfy the triangle inequality:
    #   * path_dist(a,b) = path_dist(b,a)
    #   * path_dist(a,b) + path_dist(b,c) >= path_dist(a,c)
    #   Indiscernibility is not necessary, so path_dist may not
    #   be a metric in the strict sense.
    # Output: a class with two fields:
    # * .p_var = max sum_k path_dist(a_{k-1}, a_k)^p
    #            over all strictly increasing subsequences a_k of 0,...,path_size-1
    # * .points = the maximising sequence a_k
    # Notes:
    # * if path_size == 0, the result is .p_var = -math.inf, .points = []
    # * if path_size == 1, the result is .p_var = 0,         .points = [0]

    if path_size == 0:
        return -math.inf
    elif path_size == 1:
        return 0

    s = path_size - 1
    N = 1
    while s >> N != 0:
        N += 1
    ind = [0.0] * s
    def ind_n(j, n):
        return (s >> n) + (j >> n)
    def ind_k(j, n):
        return min(((j >> n) << n) + (1 << (n-1)), s);
    max_p_var = 0.0


    run_p_var = [0.0] * path_size
    point_links = [0] * path_size

    for j in range(0, path_size):
        for n in range(1, N + 1):
            if not(j >> n == s >> n and (s >> (n-1)) % 2 == 0):
                ind[ind_n(j, n)] = max(ind[ind_n(j, n)], path_dist(ind_k(j, n), j))
        if j == 0:
            continue
        m = j - 1
        delta = 0.0
        delta_m = j
        n = 0
        while True:
            while n > 0 and m >> n == s >> n and (s >> (n-1)) % 2 == 0:
                n -= 1;
            skip = False
            if n > 0:
                iid = ind[ind_n(m, n)] + path_dist(ind_k(m, n), j)
                if delta >= iid:
                    skip = True
                elif m < delta_m:
                    delta = pow(max_p_var - run_p_var[m], 1. / p)
                    delta_m = m
                    if delta >= iid:
                        skip = True
            if skip:
                k = (m >> n) << n
                if k > 0:
                    m = k - 1
                    while n < N and (k >> n) % 2 == 0:
                        n += 1
                else:
                    break
            else:
                if n > 1:
                    n -= 1
                else:
                    d = path_dist(m, j)
                    if d >= delta:
                        new_p_var = run_p_var[m] + pow(d, p)
                        if new_p_var >= max_p_var:
                            max_p_var = new_p_var
                            point_links[j] = m
                    if m > 0:
                        while n < N and (m >> n) % 2 == 0:
                            n += 1
                        m -= 1
                    else:
                        break
        run_p_var[j] = max_p_var

    points = []
    if optim_partition:
        point_i = s
        while True:
            points.append(point_i)
            if point_i == 0:         
                break
            point_i = point_links[point_i]
        points.reverse()

    if pth_root:
        return run_p_var[-1]**(1./p), points
    return run_p_var[-1], points