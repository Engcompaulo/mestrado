import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.neural_network import MLPClassifier
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

    mlp = MLPClassifier(hidden_layer_sizes=(1024, 1),
                            activation = 'relu', 
                            solver='sgd',
                            random_state=16,
                            learning_rate_init=0.01)
    mlp.fit(x_train, y_train)
    score_test = round(mlp.score(x_test, y_test),2)
    print('ACERTOU: {}% | ERROU: {}%'.format(score_test*100,100 - score_test*100))

if __name__ == "__main__":
    main()