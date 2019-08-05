
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join

TRAIN_SIZE = 0.8

def get_data():    

    data = np.genfromtxt('./raman-spectroscopy-of-diabetes/earLobe.csv', delimiter=',')[2:,1:]
    np.random.shuffle(data)

    Y = data[:,:1].ravel()

    X = data[:,1:]
    x_transformed = []
    for x in X:
        # x = baseline_als(x)
        # x = x / x.max(axis=0)
        x_transformed.append(x)
    
    x_data_training = X[:int(len(x_transformed)*TRAIN_SIZE)]
    y_data_training = Y[:int(len(Y)*TRAIN_SIZE)]
    x_data_test = X[int(len(x_transformed)*TRAIN_SIZE):]
    y_data_test = Y[int(len(Y)*TRAIN_SIZE):]      
    
    return x_data_training, y_data_training, x_data_test, y_data_test

# Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens
def baseline_als(y, lam = 6, p = 0.05, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def main():

    x_train, y_train, x_test, y_test = get_data()

    for n in [2, 3, 5, 10, 16]:
        sfs = SFS(KNeighborsClassifier(n_neighbors=3), 
                k_features=n,       
                forward=True, 
                floating=True, 
                scoring='accuracy',
                cv=0)
        sfs = sfs.fit(x_train, y_train)

        print('\nSequential Forward Selection:')
        feat_cols = list(sfs.k_feature_idx_)
        print(feat_cols) 
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train[:, feat_cols], y_train)

        y_train_pred = knn.predict(x_train[:, feat_cols])
        print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

        y_test_pred = knn.predict(x_test[:, feat_cols])
        print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

if __name__ == "__main__":
    main()          