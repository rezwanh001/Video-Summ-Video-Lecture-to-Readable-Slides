#-*- coding: utf-8 -*-
"""
@adaptation: Md. Rezwanul Haque

@references:
    (1) http://lear.inrialpes.fr/people/potapov/med_summaries.php [ http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz ]
    (2) https://hal.inria.fr/hal-01022967/PDF/video_summarization.pdf
    (3) https://github.com/SinDongHwan/pytorch-vsumm-reinforce
"""

import numpy as np
from .cpd_nonlin import cpd_nonlin

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """
        Main interface
        
        Detect change points automatically selecting their number.
        args:
            K           :       kernel between each pair of frames in video
            ncp         :       maximum ncp
            vmax        :       special parameter
        optional args:
            lmin        :       minimum segment length
            lmax        :       maximum segment length
            desc_rate   :       rate of descriptor sampling (vmax always corresponds to 1x)

        Returns: (cps, costs)
            cps         :       best selected change-points
            costs       :       costs for 0,1,2,...,m change-points

        ---
        Note:
            * cps are always calculated in subsampled coordinates irrespective to
                desc_rate.
            * lmin and m should be in agreement.
        ---
        Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
        That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling
    
    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)
    
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, costs)
    

# ------------------------------------------------------------------------------
# Extra functions (currently not used)

def estimate_vmax(K_stable):
    """
        arg:
            K_stable        :       kernel between all frames of a stable segment
    """
    n = K_stable.shape[0]
    vmax = np.trace(centering(K_stable)/n)
    return vmax


def centering(K):
    """
        Apply kernel centering
        args:
            K               :       kernel between each pair of frames in video.
    """
    mean_rows = np.mean(K, 1)[:, np.newaxis]
    return K - mean_rows - mean_rows.T + np.mean(mean_rows)


def eval_score(K, cps):
    """ 
        Evaluate unnormalized empirical score
            (sum of kernelized scatters) for the given change-points 
    """
    N = K.shape[0]
    cps = [0] + list(cps) + [N]
    V1 = 0
    V2 = 0
    for i in range(len(cps)-1):
        K_sub = K[cps[i]:cps[i+1], :][:, cps[i]:cps[i+1]]
        V1 += np.sum(np.diag(K_sub))
        V2 += np.sum(K_sub) / float(cps[i+1] - cps[i])
    return (V1 - V2)


def eval_cost(K, cps, score, vmax):
    """ 
        Evaluate cost function for automatic number of change points selection
        args:
            K       :       kernel between all frames
            cps     :       selected change-points
            score   :       unnormalized empirical score (sum of kernelized scatters)
            vmax    :       vmax parameter
    """
    
    N = K.shape[0]
    penalty = (vmax*len(cps)/(2.0*N))*(np.log(float(N)/len(cps))+1)
    return score/float(N) + penalty