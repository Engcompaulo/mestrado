import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import decomposition
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

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size_a):
            for o in xrange(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)    

def main():

    x_train, y_train, x_test, y_test = generate_train_test_data()

    pca = decomposition.PCA(n_components=2)
    pca.fit(x_train)    
    pca_x_train = pca.transform(x_train)
    pca_x_test = pca.transform(x_test)

    mlp = MLPClassifier(hidden_layer_sizes=(1),
                            activation = 'relu', 
                            solver='sgd',
                            verbose=True,
                            max_iter=200,
                            tol=1e-4,
                            random_state=0,
                            learning_rate_init=0.1)
    mlp.fit(pca_x_train, y_train)
    score_test = round(mlp.score(pca_x_test, y_test),2)
    print('ACERTOU: {}% | ERROU: {}%'.format(score_test*100,100 - score_test*100))

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Loss Evolution', fontsize = 20)
    ax.set_xlabel('Quantidade de Iterações', fontsize = 15)
    ax.set_ylabel('Loss', fontsize = 15)
    ax.plot(mlp.loss_curve_, label='Loss Evolution')
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()