
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.neural_network import MLPClassifier
from os import listdir
from os.path import isfile, join

TRAIN_SIZE = 0.8 

def generate_train_test_data():    

    data_training_files = [f for f in listdir('./data_training') if isfile(join('./data_training', f))]
    data_test_files = [f for f in listdir('./data_test') if isfile(join('./data_test', f))]

    x_data_training = []
    y_data_training = []

    for f in data_training_files:
        matchObj = re.match(r'.*\-([0-9]{1,2})M.*', f, re.M|re.I)
        per_metanol = int(matchObj.group(1))
        x, y = np.genfromtxt('./data_training/' + f)[:,1], per_metanol > 6
        x_data_training.append(x)
        y_data_training.append(y)

    x_data_test = []
    y_data_test = []

    for f in data_test_files:
        matchObj = re.match(r'.*\_M([0-9]{1,2}).*', f, re.M|re.I)
        per_metanol = 0
        if(matchObj != None):
            per_metanol = int(matchObj.group(1))
        x, y = np.genfromtxt('./data_test/' + f)[:,1], per_metanol > 6
        x_data_test.append(x)
        y_data_test.append(y)
    
    return x_data_training, y_data_training, x_data_test, y_data_test

def main():

    x_train, y_train, x_test, y_test = generate_train_test_data()

    mlp = MLPClassifier(hidden_layer_sizes=(5, 3),
                            activation = 'relu', 
                            solver='sgd',
                            random_state=16,
                            learning_rate_init=0.001)
    mlp.fit(x_train, y_train)
    score_test = round(mlp.score(x_test, y_test),2)
    print('ACERTOU: {}% | ERROU: {}%'.format(score_test*100,100 - score_test*100))

if __name__ == "__main__":
    main()