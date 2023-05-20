import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import time
from sklearn.neighbors import KernelDensity
from gen_objects import shapes_3d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange

import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
# Or use: https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

from grasps_3d import grasps
from kde_pdf import kde_pdf

num_fingers = 4
with_normals = '_withN'

Sh = shapes_3d()
G = grasps()

num_objs = len(Sh.D.keys())

prob_min = 0.95

C_data = []

K = kde_pdf(num_fingers = num_fingers, with_normals = with_normals)
objs = K.objs

conf_matrix = []
conf_matrix = dict.fromkeys(objs)#, dict.fromkeys(objs, 0))   
for obj in objs:
    conf_matrix[obj] = dict.fromkeys(objs, 0)

M = 100
C = []
suc = 0
print('Computing confusion matrix with BC. Please wait...')
for obj in objs:
    print('Computing for object %s.'%(obj))
    X, N = Sh.D[obj][:,:3], Sh.D[obj][:,3:]

    for k in trange(M):
        P = np.ones((num_objs,)) / num_objs
        count = 0
        while 1:
            x, n = G.sample_grasp(X, N, num_fingers = num_fingers)
            v, fail = G.parameterize_grasp(x, n, normalize = False, with_normals = (False if with_normals == '_noN' else True))
            if fail:
                continue

            p_x_C = K.get_probability_allObjs(v.reshape(1,-1))

            P *= p_x_C.reshape((4,))
            P /= np.sum(P)
            count += 1

            if np.any(P > prob_min):
                break

        j = np.argmax(P)
        conf_matrix[obj][objs[j]] += 1
        if obj == objs[j]:
            suc += 1
            C.append(count)

    conf_matrix[obj].update({n: conf_matrix[obj][n] / float(M) for n in conf_matrix[obj].keys()})

print('Success rate: ' + str(float(suc)/(M*num_objs)*100.) + '%, avg. iterations: ' + str(np.mean(C)) + ' +- ' + str(np.std(C)))
C_data.append((num_fingers, with_normals, C))

CM = np.zeros((num_objs,num_objs))
O = range(num_objs)
for obj1, o1 in zip(objs, O):
    for obj2, o2 in zip(objs, O):
        CM[o1,o2] = conf_matrix[obj1][obj2]

print(CM)

plt.figure()
ax = plt.subplot()
df_cm = pd.DataFrame(CM, index = objs, columns = objs)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, cmap='Greens', annot_kws={"size": 16}, fmt='.3f') # font size
ax.set_xlabel('Predicted label', fontsize = 12)
ax.set_ylabel('True label', fontsize = 12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 90, fontsize = 10, va = 'center')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 10, va = 'center')
plt.savefig('./figures/CM_BC_objs' + str(num_objs) + '_f' + str(num_fingers) + with_normals + '.png')
plt.show()
