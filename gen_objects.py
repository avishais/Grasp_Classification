'''
Code written by:
Dr. Avishai Sintov
Robotics lab, Tel-Aviv University
Email: sintov1@tauex.tau.ac.il
May 2023
'''

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from stl import mesh # numpy-stl
from scipy.spatial import ConvexHull
import glob, os

'''
Class shapes_3d generates a point cloud for all .slt models in ./cad/ sub-directory. For each point it also generates the normal at the contact point.

Objects taken from:
https://h2t-projects.webarchiv.kit.edu/Projects/ObjectModelsWebUI/index.php?section=listAll
'''

class shapes_3d(object):
    models_dir = './cad/'

    def __init__(self):
        self.D = {}

        self.cads = glob.glob(self.models_dir + '*.stl')#[:2]
        for i in range(len(self.cads)):
            self.cads[i]  = self.cads[i][len(self.models_dir):-4]

        self.gen_shapes()

    def gen_shapes(self):
        for cad in self.cads:
            try:
                ms = mesh.Mesh.from_file(self.models_dir + cad + '.stl')
            except:
                continue
            X = []
            N = []
            for x, n in zip(ms.points, ms.normals):
                pn = np.mean(x.reshape(3,3),axis=0)
                X.append(pn)
                n /= -np.linalg.norm(n)
                N.append(n)
            X = np.array(X)
            N = np.array(N)
            X = np.concatenate((X, N), axis=1)

            self.D[cad] = X

    def plot(self, grasp = None, obj = None):

        for cad in self.cads:
            X, N = self.D[cad][:,:3], self.D[cad][:,3:]
            
            if not obj or obj == cad:
                fig = plt.figure()
                axes = mplot3d.Axes3D(fig)
                axes.plot3D(X[:,0], X[:,1], X[:,2], '.k')
                # axes.quiver(X[:,0], X[:,1], X[:,2], N[:,0], N[:,1], N[:,2], length=1, normalize=True)
                # axes.axis('equal')
            
            if grasp and cad == obj:
                X, N = grasp[0], grasp[1]
                tri = ConvexHull(X)

                axes.plot3D(X[:,0], X[:,1], X[:,2], 'or-')
                # axes.plot3D([X[-1,0], X[0,0]], [X[-1,1], X[0,1]], [X[-1,2], X[0,2]], 'r-')
                axes.plot_trisurf(X[:,0], X[:,1], X[:,2], triangles=tri.simplices, color=(0,0,0,0), edgecolor='red')
                for x, n in zip(X, N):
                    axes.quiver(x[0], x[1], x[2], 5*n[0], 5*n[1], 5*n[2], length=.1, normalize=True, color = 'magenta')

            plt.title(cad)
            # plt.axis('equal')
        plt.show()



if __name__ == "__main__":
    F = shapes_3d()
    F.plot()













