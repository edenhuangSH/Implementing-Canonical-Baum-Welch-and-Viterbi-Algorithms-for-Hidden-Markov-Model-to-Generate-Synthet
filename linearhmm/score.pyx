
from __future__ import division
import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef element_divide(double[:,:] u, double[:, :] v, double[:,:] res):
    cdef int i, j
    cdef int m, n

    m = u.shape[0]
    n = u.shape[1]

    with cython.nogil:
        for i in range(m):
            for j in range(n):
                if v[i,j]!=0:
                    res[i,j] = u[i,j] / v[i,j]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef element_multiply(double[:,:] u, double[:, :] v, double[:,:] res):
    cdef int i, j
    cdef int m, n

    m = u.shape[0]
    n = u.shape[1]

    with cython.nogil:
        for i in range(m):
            for j in range(n):
                res[i,j] = u[i,j] * v[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef matrix_multiply(double[:,:] u, double[:, :] v):
    cdef int i, j, k
    cdef int m, n, p

    m = u.shape[0]
    n = u.shape[1]
    p = v.shape[1]

    _res = np.zeros((m,p))
    cdef double[:, :] res = _res

    with cython.nogil:
        for i in range(m):
            for j in range(p):
                res[i,j] = 0
                for k in range(n):
                    res[i,j] += u[i,k] * v[k,j]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef axis_sum(double[:,:] u, int axis, double[:,:] res):
    cdef int m, n

    m = u.shape[0]
    n = u.shape[1]

    with cython.nogil:
        if axis == 1:
            for i in range(m):
                for j in range(n):
                    res[0,i] += u[i,j]
        else:
            for j in range(n):
                for i in range(m):
                    res[0,j] += u[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
def score(double[:,:] transition_probability, double[:,:] emission_probability, \
           long[:]observations,  probabilities, forward, backward):

    # declare
    cdef int k, n, m, _i
    cdef long _obs
    cdef double[:,:] _vec, _mul1, _mul2, _sum, _division, _comb, _temp
    # cdef double[:,:] probabilities

    # initialize
    k = observations.size
    n = emission_probability.shape[0]
    m = emission_probability.shape[1]

    # cannot run if declared as double[:,:] ???
    # forward = np.zeros((n,k+1))
    # cdef double[:,:] forward = _forward

    # cannot run if declared as double[:,:] ???
    # backward = np.zeros((n,k+1))
    # cdef double[:,:] backward = _backward


    # forward
    forward[:, 0] = 1/n
    for _i in range(k):
        # _vec = np.matrix(forward[:,_i])
        _vec = forward[None,:,_i]
        _obs = observations[_i]
        #_temp = _vec @ transition_probability @ np.diag(emission_probability[:,_obs])
        #forward[:,_i+1] = _temp/np.sum(_temp)
        _mul1 = matrix_multiply(_vec, transition_probability)
        _mul2 = matrix_multiply(_mul1, np.diag(emission_probability[:,_obs]))

        # sum: hardcoding output dimension of sum
        _sum = np.zeros((1,1))
        axis_sum(_mul2, 1, _sum)


        # division: propagate denominator
        _division = np.zeros((1,n))
        element_divide(_mul2, np.repeat(_sum,2,axis=1),_division)

        forward[:,_i+1] = _division

    # backward
    backward[:,-1] = 1
    for _i in range(k, 0, -1):
        # _vec = np.matrix(backward[:,_i]).transpose()
        _vec = backward[None,:,_i].T
        _obs = observations[_i-1]
#         temp = (transition_probability @ np.diag(emission_probability[:,_obs]) @ _vec).transpose()
#         backward[:,_i-1] = temp/np.sum(temp)

        _mul1 = matrix_multiply(transition_probability, np.diag(emission_probability[:,_obs]))
        _mul2 = matrix_multiply(_mul1, _vec)
        _temp = _mul2.T

        # sum: hardcoding output dimension of sum
        _sum = np.zeros((1,1))
        axis_sum(_temp, 1, _sum)


        # division: propagate denominator
        _division = np.zeros((1,n))
        element_divide(_temp, np.repeat(_sum,2,axis=1),_division)

        backward[:,_i-1] = _division

    # combine forward and backward
    _comb = np.zeros((n,k+1))
    element_multiply(forward, backward, _comb)

    # get the sum along the 0 axis
    _sum = np.zeros((1,k+1))
    axis_sum(_comb, 0, _sum)
    _temp = np.repeat(_sum, 2 , axis=0)

    # get probabilites by dividing the combination by the sum
    element_divide(_comb, _temp, probabilities)

    # return probabilities, forward, backward
