'''
Code written by:
Dr. Avishai Sintov
Robotics lab, Tel-Aviv University
Email: sintov1@tauex.tau.ac.il
May 2023
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import time
import glob, os
from gen_objects import shapes_3d
from tqdm import tqdm
from grasps_3d import grasps

Sh = shapes_3d()
G = grasps()

# Generate a set of M grasp samples for each object
M = 10000

# Determine number of fingers in a grasp sample
num_fingers = 4

# To include normal at the grasp ('_withN') or not ('_noN')
with_normals = '_withN'

objs = list(Sh.D.keys())

CLS = 0
Input = []
Labels = []
for obj in objs:
    print ('Sampling from object ' + obj + '...')
    X, N = Sh.D[obj][:,:3], Sh.D[obj][:,3:]

    i = 0
    pbar = tqdm(total=M)
    while i < M:
        x, n = G.sample_grasp(X, N, num_fingers = num_fingers)
        
        v, fail = G.parameterize_grasp(x, n, normalize = False, with_normals = (False if with_normals == '_noN' else True))
        if fail:
            continue

        Input.append(v)
        Labels.append(CLS)
        i += 1
        pbar.update(1)
    pbar.close()

    CLS += 1

Input, Labels = np.array(Input), np.array(Labels)
fileName = './data/id_data_' + str(num_fingers) + with_normals + '.pkl'
with open(fileName, 'wb') as H:
    pickle.dump([Input, Labels, objs], H)

print('A set of %d grasps for %d objects was saved in file: %s'%(Input.shape[0], len(objs), fileName))



