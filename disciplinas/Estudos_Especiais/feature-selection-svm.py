
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import decomposition
from sklearn.svm import SVC
from os import listdir
from os.path import isfile, join

TRAIN_SIZE = 0.8

def get_data():    

    # data = np.genfromtxt('./raman-spectroscopy-of-diabetes/earLobe.csv', delimiter=',')[2:,1:]
    data = np.genfromtxt('./raman-spectroscopy-of-candida-fungo/candida.csv', delimiter=',')

    true_data = np.array([d for d in data if d[0] == 0])
    false_data = np.array([d for d in data if d[0] == 1])

    np.random.shuffle(true_data)
    np.random.shuffle(false_data)

    Y_true = true_data[:,:1].ravel()
    Y_false = false_data[:,:1].ravel()

    X_true = []
    for x in true_data[:,1:]:
        x = baseline_als(x)
        np.random.shuffle(x)
        # x = x / x.max(axis=0)
        X_true.append(x)
            
    X_false = []
    for x in false_data[:,1:]:
        x = baseline_als(x)
        np.random.shuffle(x)
        # x = x / x.max(axis=0)
        X_false.append(x)
    
    x_data_training = np.array(X_true[:int(len(X_true)*TRAIN_SIZE)] + X_false[:int(len(X_false)*TRAIN_SIZE)])
    y_data_training = np.concatenate((Y_true[:int(len(Y_true)*TRAIN_SIZE)], Y_false[:int(len(Y_false)*TRAIN_SIZE)]))
   
    x_data_test = np.array(X_true[int(len(X_true)*TRAIN_SIZE):] + X_false[int(len(X_false)*TRAIN_SIZE):])
    y_data_test = np.concatenate((Y_true[int(len(Y_true)*TRAIN_SIZE):], Y_false[int(len(Y_false)*TRAIN_SIZE):]))
    
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
        sfs = SFS(SVC(gamma='auto'), 
                k_features=n,       
                forward=True, 
                floating=True, 
                scoring='accuracy',
                cv=0)
        sfs = sfs.fit(x_train, y_train)

        print('\nSequential Floating Forward Selection: ', n)
        feat_cols = list(sfs.k_feature_idx_)
        print(feat_cols) 
        
        svc = SVC(gamma='auto')
        svc.fit(x_train[:, feat_cols], y_train)

        y_train_pred = svc.predict(x_train[:, feat_cols])
        print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

        y_test_pred = svc.predict(x_test[:, feat_cols])
        print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

        print(confusion_matrix(y_test, y_test_pred))
        print(classification_report(y_test, y_test_pred))

        if n == 2:
            fig, axs = plt.subplots(2)
            fig.suptitle("SFS(SVM) Scatter Plot", fontsize='small')
            axs[0].scatter(x_train[:, feat_cols[0]], x_train[:, feat_cols[1]], marker='o', c=y_train, s=25, edgecolor='k')
            axs[1].scatter(x_test[:, feat_cols[0]], x_test[:, feat_cols[1]], marker='o', c=y_test, s=25, edgecolor='k')

            plt.show()



if __name__ == "__main__":
    main()          