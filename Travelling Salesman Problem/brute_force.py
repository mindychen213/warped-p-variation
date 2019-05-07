import itertools
import math

def brute_force(n, distances):
    minLength = math.inf
    minTour = []

    for tour in itertools.permutations(list(range(1,n))):
        fr = 0
        length = 0
        count = 0
        while count<n-1:
            to = tour[count]
            length += distances[fr][to]
            fr = to
            count += 1
        length += distances[fr][0]
        if length < minLength:
            minLength = length
            minTour = tour
    minTour = (0,) + minTour + (0,)
    print('Shortest tour is: ', minTour)
    print('It has a length of: ', minLength , ' km')