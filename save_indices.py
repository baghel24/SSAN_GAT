 # -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio

def Sampling(groundtruth): 
    #  Dictionaries to store indices of each class for different datasets.
    labeled = {}
    test    = {}
    valid   = {}
    all     = {}
    m = max(groundtruth)
    # Lists to store combined indices from all classes for each dataset. 
    labeled_indices = []
    test_indices    = []
    valid_indices   = []
    all_indices     = []
    for i in range(m+1):
        indices = [j for j, x in enumerate(groundtruth.ravel().tolist()) if x == i]
        if i != 0:
            np.random.shuffle(indices)
            all[i]     = indices
            test[i]    = indices[200:]
            valid[i]   = indices[100:200]
            labeled[i] = indices[:100]
            labeled_indices += labeled[i]
            valid_indices   += valid[i]
            test_indices    += test[i]            
            all_indices     += all[i]

    np.random.shuffle(labeled_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(all_indices)
    return labeled_indices, test_indices, valid_indices,all_indices

mat_gt = sio.loadmat("Pavia_gtt.mat")
label  = mat_gt['pavia_gt']

# Reshaping the ground truth data for easier manipulation.
GT     = label .reshape(np.prod(label.shape[:2]),)

labeled_indices, test_indices, valid_indices,all_indices= Sampling(GT)

np.save('labeled_index.npy', labeled_indices)
np.save('valid_index.npy', valid_indices)
np.save('test_index.npy', test_indices)
np.save('all_index.npy', all_indices)




 


