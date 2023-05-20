import numpy as np 
import pickle
import time
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# Class for implementing a Bayesian Classifier

class kde_pdf(object):
    bandwidth = 0.5

    def __init__(self, num_fingers = 4, with_normals = '_withN'):
        self.num_fingers = num_fingers
        self.with_normals = with_normals

        self.load_models()

    def create_models(self):
        print('Generating KDE models for all objects with bandwith %.2f...'%self.bandwidth)

        # load the dataset
        with open('./data/id_data_' + str(self.num_fingers) + self.with_normals + '.pkl', 'rb') as H:
            X, y, self.objs = pickle.load(H)

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.models = dict()
        for i in range(len(self.objs)):
            obj = self.objs[i]

            obj_data = X[y == i, :]

            self.models[obj] = KernelDensity(bandwidth = self.bandwidth, kernel='gaussian')
            self.models[obj].fit(obj_data)

        with open('./data/kde_model_objs' + str(len(self.objs)) + '_f' + str(self.num_fingers) + self.with_normals + '_v2.pkl', 'wb') as H:
            pickle.dump([self.models, self.objs, self.scaler], H)

        print('Finished generating KDE models.')

    def load_models(self, num_objs = 4):
        try: 
            with open('./data/kde_model_objs' + str(num_objs) + '_f' + str(self.num_fingers) + self.with_normals + '_v2.pkl', 'rb') as H:
                self.models, self.objs, self.scaler = pickle.load(H)
            print("Data loaded.")
        except:
            self.create_models()

    def get_probability(self, x, obj):
        x = self.scaler.transform(x.reshape(1,-1))
        return np.exp(self.models[obj].score_samples(x))

    # Output likelihood p(observation|class) for all classes
    def get_probability_allObjs(self, x):
        x = self.scaler.transform(x.reshape(1,-1))
        p = []
        for obj in self.objs:
            p.append( np.exp(self.models[obj].score_samples(x)) )

        return np.array(p)