#-*- coding: utf-8 -*-
"""
@adaptation: Md. Rezwanul Haque

@references:
    (1) http://lear.inrialpes.fr/people/potapov/med_summaries.php [ http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz ]
    (2) https://hal.inria.fr/hal-01022967/PDF/video_summarization.pdf
    (3) https://github.com/SinDongHwan/pytorch-vsumm-reinforce
"""

import numpy as np

def calc_scatters(K):
    """
        Calculate scatter matrix:
        args:
            K               :       square kernel matrix
        returns:
            scatters[i,j]   :       {scatter of the sequence with starting frame i and ending frame j}
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1); # TODO: use the fact that K - symmetric

    scatters = np.zeros((n, n))

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1,1))
    j = np.arange(n).reshape((1,-1))
    scatters = (K1[1:].reshape((1,-1))-K1[:-1].reshape((-1,1))
                - (diagK2[1:].reshape((1,-1)) + diagK2[:-1].reshape((-1,1)) - K2[1:,:-1].T - K2[:-1,1:]) / ((j-i+1).astype(float) + (j==i-1).astype(float)))
    scatters[j<i]=0

    return scatters

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, \
                verbose=True, out_scatters=None):
    """ 
        Change point detection with dynamic programming
        args:
            K           :       square kernel matrix
            ncp         :       number of change points to detect (ncp >= 0)
            lmin        :       minimal length of a segment
            lmax        :       maximal length of a segment
            backtrack   :       when False - only evaluate objective scores (to save memory)

        returns: (cps, scores)
            cps         :       detected array of change points: 
                                    mean is thought to be constant on ( cps[i], cps[i+1] )
            scores      :       obj_vals - values of the objective function for 0..m changepoints
    """
    m = int(ncp)  # prevent numpy.int64

    (n, n1) = K.shape
    assert(n == n1), "Kernel matrix awaited."

    assert(n >= (m + 1)*lmin)
    assert(n <= (m + 1)*lmax)
    assert(lmax >= lmin >= 1)

    if verbose:
        #print "n =", n
        print("Precomputing scatters...")
    J = calc_scatters(K)

    if out_scatters != None:
        out_scatters[0] = J

    if verbose:
        print("Inferring best change points...")
    # I[k, l] - value of the objective for k change-points and l first frames
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]

    if backtrack:
        # p[k, l] --- "previous change" --- best t[k] when t[k+1] equals l
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)

    for k in range(1,m+1):
        for l in range((k+1)*lmin, n+1):
            tmin = max(k*lmin, l-lmax)
            tmax = l-lmin+1
            c = J[tmin:tmax,l-1].reshape(-1) + I[k-1, tmin:tmax].reshape(-1)
            I[k,l] = np.min(c)
            if backtrack:
                p[k,l] = np.argmin(c)+tmin

    # Collect change points
    cps = np.zeros(m, dtype=int)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores