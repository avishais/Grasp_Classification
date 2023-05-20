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
from gen_objects import shapes_3d
from grasps_3d import grasps
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tqdm import tqdm, trange

import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
# Or use: https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

num_fingers = 4
with_normals = '_withN'

Sh = shapes_3d()
G = grasps()
num_objs = len(Sh.D.keys())

prob_min = 0.85 # Lambda_s

C_data = []

print('Computing with num_fingers %d and %s.'%(num_fingers, with_normals))
    
with open('./tf_models/tf_model_' + str(num_fingers) + with_normals + '.pkl', 'rb') as R:
    H, objs, acc, scaler = pickle.load(R)
print('Using model of size: ', H)

n_features = len(scaler.mean_)
clf = Sequential()
clf.add(Dense(H[0], activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
for j in range(1,len(H)):
    clf.add(Dense(H[j], activation='relu', kernel_initializer='he_normal'))
clf.add(Dense(len(objs), activation='softmax'))
checkpoint_path = './tf_models/model_' + str(num_fingers) + with_normals + '.ckpt'
clf.load_weights(checkpoint_path)

conf_matrix = dict.fromkeys(objs)
for obj in objs:
    conf_matrix[obj] = dict.fromkeys(objs, 0)

M = 100
C = []
suc = 0
print('Computing confusion matrix with IC. Please wait...')
for obj in objs:
    print('Computing for object %s.'%(obj))
    X, N = Sh.D[obj][:,:3], Sh.D[obj][:,3:]
    
    for k in trange(M):
        P = np.zeros((num_objs,))
        p_all = 1./num_objs
        count = 0
        while count <= 1 or p_all < prob_min: #
            x, n = G.sample_grasp(X, N, num_fingers = num_fingers)
            v, fail = G.parameterize_grasp(x, n, normalize = False, with_normals = (False if with_normals == '_noN' else True))
            if fail:
                continue
            
            v = scaler.transform(v.reshape(1,-1))
            p = clf.predict(v.reshape(1,-1), verbose=0)[0]
            j = np.argmax(p)
            
            P[j] += p[j]
            if P[j] < 0.6:
                p_all = P[j]
            else:
                p_all = P[j]/sum(P)

            count += 1
            if p_all < prob_min and count > 50:
                P = np.zeros((num_objs,))
                p_all = 1./num_objs
                count = 0

        j = np.argmax(P)
        conf_matrix[obj][objs[j]] += 1
        if obj == objs[j]:
            suc += 1
            C.append(count)

    conf_matrix[obj].update({n: conf_matrix[obj][n] / float(M) for n in conf_matrix[obj].keys()})

print('Success rate: ' + str(float(suc)/(M*num_objs)*100.) + '%, avg. iterations: ' + str(np.mean(C)) + ' +- ' + str(np.std(C)))
print(conf_matrix)

CM = np.zeros((num_objs,num_objs))
O = range(num_objs)
for obj1, o1 in zip(objs, O):
    for obj2, o2 in zip(objs, O):
        CM[o1,o2] = conf_matrix[obj1][obj2]
print(CM)

plt.figure()
ax= plt.subplot()
df_cm = pd.DataFrame(CM, index = objs, columns = objs)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, cmap='Greens', annot_kws={"size": 16}, fmt='.3f') # font size
ax.set_xlabel('Predicted label', fontsize = 12)
ax.set_ylabel('True label', fontsize = 12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 90, fontsize = 10, va = 'center')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 10, va = 'center')
plt.savefig('./figures/CM_IC_objs' + str(num_objs) + '_f' + str(num_fingers) + with_normals + '.png')
plt.show()






