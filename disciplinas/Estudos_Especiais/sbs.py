
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  

def get_data():    
    data = np.genfromtxt('./raman-spectroscopy-of-diabetes/earLobe.csv', delimiter=',')[2:,1:]

    Y = data[:,:1].ravel()
    X = data[:,1:]
    
    return X, Y

def main():

    X, Y = get_data()

    knn = KNeighborsClassifier(n_neighbors=4)

    sbs = SFS(knn, 
            k_features=3, 
            forward=False, 
            floating=False, 
            scoring='accuracy',
            cv=0)
    sbs = sbs.fit(X, Y)

    print('\nSequential Backward Selection (k=3):')
    print(sbs.k_feature_idx_)
    print('CV Score:')
    print(sbs.k_score_)

    print(pd.DataFrame.from_dict(sbs.get_metric_dict()).T)

if __name__ == "__main__":
    main()          