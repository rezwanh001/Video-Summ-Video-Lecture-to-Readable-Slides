#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque

@references:
    (1) http://lear.inrialpes.fr/people/potapov/med_summaries.php [ http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz ]
    (2) https://hal.inria.fr/hal-01022967/PDF/video_summarization.pdf
    (3) https://github.com/SinDongHwan/pytorch-vsumm-reinforce
"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import math
import cv2
import matplotlib.pyplot as plt
from KTS.cpd_nonlin import cpd_nonlin
from KTS.cpd_auto import cpd_auto
from generate_dataset import Generate_Dataset
from KTS.config import CONFIG
from KTS.utils import LOG_INFO

np.random.seed(CONFIG.SEED)

def gen_data(n, m, d=1):
    """
        Generates data with change points
        args:
            n       :       number of samples
            m       :       number of change-points
            d       :       d-dimensional signal 
        returns:
            X       :       data array (n X d)
            cps     :       change-points array, including 0 and n
        
    """
    
    # Select changes at some distance from the boundaries
    cps = np.random.permutation(int(n*3/4)-1)[0:m] + 1 + n/8
    cps = [round(x) for x in cps]
    cps = np.sort(cps)
    cps = [0] + list(cps) + [n]
    mus = np.random.rand(m+1, d)*(m/2)  # make sigma = m/2
    X = np.zeros((n, d))
    for k in tqdm(range(m+1)):
        X[cps[k]:cps[k+1], :] = mus[k, :][np.newaxis, :] + np.random.rand(cps[k+1]-cps[k], d)
    return (X, np.array(cps))
    
if __name__ == "__main__":
    plt.ioff()
    # ===============================================================
    print ("Test 1: 1-dimensional signal")
    plt.figure("Test 1: 1-dimensional signal")
    n = 1000
    m = 10
    (X, cps_gt) = gen_data(n, m)
    print ("Ground truth:", cps_gt)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_nonlin(K, m, lmin=1, lmax=10000)
    print ("Estimated:", cps)
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')
    plt.show()
    print ("="*79)

    # ===============================================================
    print ("Test 2: multidimensional signal")
    plt.figure("Test 2: multidimensional signal")
    n = 1000
    m = 20
    (X, cps_gt) = gen_data(n, m, d=50)
    print ("Ground truth:", cps_gt)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_nonlin(K, m, lmin=1, lmax=10000)
    print ("Estimated:", cps)
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')
    plt.show()
    print ("="*79)

    # ===============================================================
    print ("Test 3: automatic selection of the number of change-points")
    plt.figure("Test 3: automatic selection of the number of change-points")
    (X, cps_gt) = gen_data(n, m)
    print ("Ground truth: (m=%d)" % m, cps_gt)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, 2*m, 1)
    print ("Estimated: (m=%d)" % len(cps), cps)
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')
    plt.show()
    print ("="*79)


    ###=====================================================================
    print ("Test 4: Frames: automatic selection of the number of change-points")
    # plt.figure("Test 4: Frames: automatic selection of the number of change-points")

    video_path = CONFIG.VIDEO_PATH 
    output_path = CONFIG.OUTPUT_PATH 
    frame_dir = CONFIG.FRAME_DIR 
    train_data = CONFIG.TRAIN_DATA ### flag: this is for extraction frames 
    plot_fig = CONFIG.PLOT_FIG
    gen = Generate_Dataset(video_path, output_path,frame_dir, train_data)
    gen.generate_dataset(plot_fig)
    gen.h5_file.close()
    LOG_INFO("="*79)
