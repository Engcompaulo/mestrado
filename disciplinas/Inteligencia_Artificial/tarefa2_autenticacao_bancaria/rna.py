
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

TRAIN_SIZE = 0.8 

def generate_train_test_data():    
    my_data = np.genfromtxt('./docs/dados_autent_bancaria.txt', delimiter=',')
    np.random.shuffle(my_data)

    true_data = [d for d in my_data if d[4] == 0]
    false_data = [d for d in my_data if d[4] == 1]

    train_data = np.array(true_data[:int(len(true_data)*TRAIN_SIZE)] + false_data[:int(len(false_data)*TRAIN_SIZE)])
    test_data = np.array(true_data[int(len(true_data)*TRAIN_SIZE):] + false_data[int(len(false_data)*TRAIN_SIZE):])

    X_train, y_train = train_data[:,:4], train_data[:,4]
    X_test, y_test = test_data[:,:4], test_data[:,4]
    
    return X_train, y_train, X_test, y_test

def main():

    X_train, y_train, X_test, y_test = generate_train_test_data()

    mlp = MLPClassifier(hidden_layer_sizes=(4, 1),
                            activation = 'relu', 
                            solver='sgd', 
                            verbose=False, 
                            tol=1e-4, 
                            random_state=32,
                            learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    score_test = round(mlp.score(X_test, y_test),2)
    print('ACERTOU: {}% | ERROU: {}%'.format(score_test*100,100 - score_test*100))

if __name__ == "__main__":
    main()