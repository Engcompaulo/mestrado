
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  

def get_data():    
    data = np.genfromtxt('./raman-spectroscopy-of-diabetes/earLobe.csv', delimiter=',')[2:,1:]

    Y = data[:,:1].ravel()
    X = data[:,1:]
    
    return X, Y

def main():

    X, Y = get_data()

    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    pca_x = pca.transform(X)
    print(pca.explained_variance_ratio_)

    knn = KNeighborsClassifier(n_neighbors=4)

    sfs = SFS(knn, 
            k_features=3,       
            forward=True, 
            floating=True, 
            scoring='accuracy',
            cv=0)
    sfs = sfs.fit(X, Y)

    print('\nSequential Forward Selection (k=3):')
    print(sfs.k_feature_idx_)
    print('CV Score:')
    print(sfs.k_score_)

    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)

if __name__ == "__main__":
    main()          