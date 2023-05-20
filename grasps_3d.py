
import numpy as np 
import matplotlib.pyplot as plt 
from gen_objects import shapes_3d
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import time
from scipy.linalg import null_space

color = 'yrbkmcgw'

class grasps(object):
    m = 10 # size of friction cone

    def __init__(self, ni = None):
        self.ni = ni

    def sample_grasp(self, X, N, num_fingers = 3):

        k = np.random.choice(X.shape[0], size=(num_fingers,), replace = False)
        k = np.sort(k)
        x = X[k,:]
        n = N[k,:]

        return x, n

    def friction_cone(self, n):
        # Calculates the m primitive wrenches constructing the friction cone around normal n.

        if not self.ni:
            return n
        r = self.ni
        m = self.m

        tz = np.arctan2(n[1], n[0])
        ty = -np.arctan2(n[2], np.sqrt(n[0]**2+n[1]**2))
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0],[np.sin(tz), np.cos(tz), 0],[0, 0, 1]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)],[0, 1, 0],[-np.sin(ty), 0, np.cos(ty)]])

        t = np.linspace(0, 2*np.pi,m+1).reshape(1,-1)  # the angles of the edges around the normal constructing the friction cone
        vc = np.concatenate((np.zeros((1,m+1)), r*np.cos(t), r*np.sin(t)), axis=0)
        vc = np.dot(Rz, np.dot(Ry, vc))
        vc = vc.transpose() + n.reshape(1,-1)
        for v in vc:
            v /= np.linalg.norm(v)
        
        # from mpl_toolkits import mplot3d
        # figure = plt.figure()
        # axes = mplot3d.Axes3D(figure)
        # axes.quiver(0,0,0, n[0], n[1], n[2], length=.1, normalize=True)
        # for v in vc:
        #     axes.quiver(0, 0, 0, v[0], v[1], v[2], length=.1, normalize=True, color = 'red')
        # plt.axis('equal')
        # plt.show()

        return vc

    def get_wrench(self, X, N):

        W = []
        for x, n in zip(X, N):
            if not self.ni:
                t = np.cross(x, n)
                w = np.array([n[0], n[1], n[2], t[0], t[1], t[2]])
                W.append(w)
            else:
                V = self.friction_cone(n)
                for v in V:
                    t = np.cross(x, v)
                    w = np.array([v[0], v[1], v[2], t[0], t[1], t[2]])
                    W.append(w)
        return np.array(W)

    def gen_CH(self, X, N):
        self.W = self.get_wrench(X, N)
        try:
            self.hull = ConvexHull(self.W)
        except:
            self.hull = None
        return self.hull

    def cnormal(self, M):
        # INPUT - A m-by-n matrix M, where m vectors in the n-d. (m=6,n=6)
        # Output - The normal of the n-d plane constructed by the m points.

        n = null_space(np.concatenate((M, -np.ones((6,1))), axis=1))
        if n.shape[1] > 1:
            if np.abs(n[6,0]) < np.abs(n[6,1]):
                n = n[:,1]
            else:
                n = n[:,0]

        d = n[6]/np.linalg.norm(n[:6])
        n = n[:6] / np.linalg.norm(n[:6])
        return n.reshape(-1,), d

    def cross6d(self, M):
        H = []
        for i in range(1,M.shape[0]):
            H.append(M[i,:] - M[0,:])
        H = np.array(H)

        v = []
        for i in range(0, M.shape[0]):
            R = np.concatenate((H[:,:i], H[:,i+1:]), axis=1)
            v.append((-1)**(i+2) * np.linalg.det(R))
        
        return v / np.linalg.norm(v)

    def check_origin(self):
        k = self.hull.simplices
        dt = self.W#[self.hull.vertices]
        self.u = np.sum(self.W, axis=0) / dt.shape[0]

        Q = 1e50
        for i in range(k.shape[0]):
            w = self.W[k[i,:],:]
            N, d = self.cnormal(w)

            Du = np.dot(N, self.u) - d

            if np.sign(Du * -d) == -1 or np.abs(d) <= 1e-2:
                return False, 0

            if np.abs(d) < Q:
                Q = np.abs(d)

        return True, Q
 
    def plot_CH(self, show = True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in self.hull.simplices:
            plt.plot(self.W[i,0], self.W[i,1], self.W[i,2], 'or')

            x = self.W[i, 0]
            y = self.W[i, 1]
            z = self.W[i, 2]
            verts = [list(zip(x,y,z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=color[np.random.randint(8)], alpha = 0.5))

        ax.plot3D(self.W[:,0], self.W[:,1], self.W[:,2], '.k')
        ax.plot3D([0],[0],[0], 'xm')
        try:
            ax.plot3D([self.u[0]], [self.u[1]], [self.u[2]], 'xk')
        except:
            pass
        
        if show:
            plt.show()

    def heron_triangle_area(self, X):

        a = np.linalg.norm(X[0,:]-X[1,:])
        b = np.linalg.norm(X[0,:]-X[2,:])
        c = np.linalg.norm(X[1,:]-X[2,:])

        return np.sqrt( (a+b+c) * (-a+b+c) * (a-b+c) * (a+b-c) ) / 4

    def normal2triangle(self, X):
        v1 = X[1] - X[0]
        v2 = X[2] - X[0]
        return (np.cross(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def grasp_stat(self, X, N):

        tri = ConvexHull(X)
        ts = tri.simplices
            
        A = tri.area
        V = tri.volume
        return A, V

    def parameterize_grasp(self, X, N, normalize = True, with_normals = True):
        # Currently works for n = 3, 4, 5 !!!!!!!!!!!!!
        n = X.shape[0] # Number of fingers
        FV = []
        ns = []

        if n == 2:
            FV.append( np.linalg.norm(X[1]-X[0]) )
            return np.array(FV), False

        if n > 3:
            try:
                tri = ConvexHull(X)
            except:
                return np.zeros((5*n-6,)), True
            ts = tri.simplices
            if ts.shape[0] < n:
                return np.zeros((5*n-6,)), True
        else:
            ts = np.array([[0,1,2]])

        A = [self.heron_triangle_area(X[t, :]) for t in ts]
        if normalize:
            k = np.sum(A)**0.5
            X /= k # Normalize the surface area of the grasp polytop to 1
            A /= k

        imn = np.argmax(A)
        A = np.roll(A, -imn)
        ts = np.roll(ts, -imn, axis = 0)

        cur_triangle = np.copy(ts[0])
        Xtri = np.copy(X[cur_triangle])
        Ntri = np.copy(N[cur_triangle])
        D = [np.linalg.norm(Xtri[i]-Xtri[i+1]) for i in range(2)]
        D.append(np.linalg.norm(Xtri[-1]-Xtri[0]))

        imn = np.argmax(D)
        D = np.roll(D, -imn)
        Xtri = np.roll(Xtri, -imn, axis = 0)
        Ntri = np.roll(Ntri, -imn, axis = 0)
        cur_triangle = np.roll(cur_triangle, -imn, axis=0)
        node_list = np.copy(cur_triangle)

        for j in range(n-2):
            # Polygon angles
            FV.append( np.arccos(np.dot(Xtri[-1]-Xtri[0],Xtri[1]-Xtri[0])/(np.linalg.norm(Xtri[-1]-Xtri[0])*np.linalg.norm(Xtri[1]-Xtri[0]))) )
            FV.append( np.arccos(np.dot(Xtri[0]-Xtri[1],Xtri[2]-Xtri[1])/(np.linalg.norm(Xtri[0]-Xtri[1])*np.linalg.norm(Xtri[2]-Xtri[1]))) )
            
            n_tri = self.normal2triangle(Xtri) # Normal to first triangle
            ns.append(n_tri)

            if j > 0:
                FV.append(np.dot(n_tri, n_prev))
                g = [np.any(b == cur_triangle[1]) * np.any(b == cur_triangle[2]) for b in ts]
                if np.any(ts[g][1] == cur_triangle[0]) * np.any(ts[g][1] == cur_triangle[1]) * np.any(ts[g][1] == cur_triangle[2]):
                    next_triangle = np.copy(ts[g][0])
                else:
                    next_triangle = np.copy(ts[g][1])
                k = next_triangle[[b != cur_triangle[1] and b != cur_triangle[2] for b in next_triangle]][0]
                cur_triangle = np.array([cur_triangle[1], cur_triangle[2], k])
            else:
                FV.append(D[0]) # Polygon lengths
                if n > 3:
                    g = [np.any(b == cur_triangle[0]) * np.any(b == cur_triangle[1]) for b in ts]
                    next_triangle = np.copy(ts[g][1])
                    cur_triangle[2] = next_triangle[np.not_equal(next_triangle, cur_triangle[0]) * np.not_equal(next_triangle, cur_triangle[1])][0]
            
            if n > 3:
                node_list = np.append(node_list, cur_triangle[2])

                Xtri = np.copy(X[cur_triangle])
                Ntri = np.copy(N[cur_triangle])
                n_prev = np.copy(n_tri)
            else:
                n_edge = Xtri[1]-Xtri[0]
                n_edge /= np.linalg.norm(n_edge)
                ns.append(n_edge)

        if with_normals:
            node_list = np.array(range(n))# np.unique(node_list)
            for j in node_list:
                FV.append( np.arccos(np.dot(ns[0], N[j])) )
                FV.append( np.arccos(np.dot(ns[1], N[j])) )

        return np.array(FV), False

    def norm_grasp(self, X):
        # Normalize the circumference of the grasp polygon to 1
        l = [np.linalg.norm(X[i]-X[i-1]) for i in range(1, X.shape[0])]
        l.append(np.linalg.norm(X[0]-X[-1]))
        X /= np.sum(l)

        return X

    def get_object_data(self, num_fingers = 3):

        Sh = shapes_3d()
        M = 2000
        for obj in Sh.D.keys():

            X, N = Sh.D[obj][:,:3], Sh.D[obj][:,3:]

            try:
                with open('./data/data_' + obj + '_' + str(self.ni) + '_' + str(num_fingers) + '.pkl', 'rb') as H:
                    F, S, Qdata = pickle.load(H)
                F, S = list(F), list(S)
                f, s = len(F), len(S)   
                if f == M and s == M:
                    continue             
            except IOError:
                print("File not accessible")
                s = 0
                f = 0
                S = []
                F = []
                Qdata = []

            T = []
            i = 0
            while s < M or f < M:
                x, n = self.sample_grasp(X, N, num_fingers=num_fingers)
                
                # st = time.time()
                
                if not self.gen_CH(x, n):
                    continue

                try:
                    fc, Q = self.check_origin()
                except:
                    continue

                # T.append(time.time() - st)
                v, fail = self.parameterize_grasp(x, n)
                if fail:
                    continue

                if fc and s < M:
                    Qdata.append((v, Q))
                    S.append(v)
                    s += 1
                if not fc and f < M:
                    F.append(v)
                    f += 1
                print (obj, s, f)

                if i % 1000 == 0:
                    Fs = np.array(F)
                    Ss = np.array(S)
                    with open('./data/data_' + obj + '_' + str(self.ni) + '_' + str(num_fingers) + '.pkl', 'wb') as H:
                        pickle.dump([Fs, Ss, Qdata], H)
                i += 1

            F = np.array(F)
            S = np.array(S)
            print (F.shape, S.shape)
            with open('./data/data_' + obj + '_' + str(self.ni) + '_' + str(num_fingers) + '.pkl', 'wb') as H:
                pickle.dump([F, S, Qdata], H)


if __name__ == "__main__":
    obj = 'Shampoo'
    Sh = shapes_3d()  # Sh has the data for all objects in the set
    X, N = Sh.D[obj][:,:3], Sh.D[obj][:,3:] # Get data of object 'obj'
    
    # Parameterize a random grasp
    num_fingers = 4 # Set a 4-finger grasp
    G = grasps(ni = 0.7) # Initialize a grasp class with friction coefficient ni
    x, n = G.sample_grasp(X, N, num_fingers=num_fingers) # Sample grasp of positions (x) and normals (n) at the points
    print(G.parameterize_grasp(x, n, with_normals=False)) # Print the parameterization vector


# G.get_object_data(num_fingers=num_fingers)
# exit(1)

# 
# T = 0
# for _ in range(1):
#     st = time.time()
#     fc = False
#     while not fc:
#         x, n = G.sample_grasp(X, N, num_fingers=num_fingers)

#         while not G.gen_CH(x, n):
#             x, n = G.sample_grasp(X, N, num_fingers=num_fingers)
#         fc, _ = G.check_origin()
    
#     T += time.time() - st
#     print G.parameterize_grasp(x, n)
# print T / 100
# G.plot_CH(show = False)
# Sh.plot(grasp = [x, n], obj = obj)

    