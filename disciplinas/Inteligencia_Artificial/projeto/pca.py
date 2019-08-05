import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import decomposition
from os import listdir
from os.path import isfile, join

def generate_train_test_data():    

    data_training_files = [f for f in listdir('./data_training') if isfile(join('./data_training', f))]
    data_test_files = [f for f in listdir('./data_test') if isfile(join('./data_test', f))]

    x_data_training = []
    y_data_training = []

    for f in data_training_files:
        matchObj = re.match(r'.*\-([0-9]{1,2})M.*', f, re.M|re.I)
        per_metanol = int(matchObj.group(1))
        x = np.genfromtxt('./data_training/' + f)[:,1]
        x = baseline_als(x)
        x = x / x.max(axis=0)
        y = per_metanol > 10
        x_data_training.append(x)
        y_data_training.append(y)

    x_data_test = []
    y_data_test = []

    for f in data_test_files:
        matchObj = re.match(r'.*\_M([0-9]{1,2}).*', f, re.M|re.I)
        per_metanol = 0
        if(matchObj != None):
            per_metanol = int(matchObj.group(1))
        x = np.genfromtxt('./data_test/' + f)[:,1]
        x = baseline_als(x)
        x = x / x.max(axis=0)
        y = per_metanol > 10
        x_data_test.append(x)
        y_data_test.append(y)
    
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

    x_train, y_train, x_test, y_test = generate_train_test_data()

    pca = decomposition.PCA(n_components=2)
    pca.fit(x_train)
    pca_x_train = pca.transform(x_train)
    print(pca.explained_variance_ratio_)

    principalDf = pd.DataFrame(data = pca_x_train, columns = ['principal component 1', 'principal component 2'])
    resultDf = pd.DataFrame(data = y_train, columns = ['target'])
    finalDf = pd.concat([principalDf, resultDf], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [True, False]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

    plt.show()  

if __name__ == "__main__":
    main()