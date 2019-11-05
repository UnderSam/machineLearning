import operator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

## input_x -> row data of x_
## features -> X_train
## labels -> y_train
## k -> k nearest
def classify(input_x, features, labels, k):
    distances = []
    for trainIdx,train in enumerate(features):
        #Minkowski default n is 2 and is equal to euclidean distance
        dist = np.sqrt(np.sum((train - input_x) ** 2))
        distances.append((trainIdx,dist,1/dist))
    #sort distance list by lambda function
    distances.sort(key=lambda x: x[1])
    count = []
    
    #voting result with list to find the class
    for i in range(k):
        count.append({"label":labels.item(distances[i][0]),"weight":distances[i][1]})
    print(count)
    return 0

def predict(X, k):
    y = []
    for i in X:
        y.append(classify(i, X_train, y_train, k))
    return y


if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0,
                                                        stratify=iris.target)

    test_data_acc = []
    for i in range(1, 3):
        y_predict_test_data = predict(X_test, i)
        test_data_acc.append(accuracy_score(y_test, y_predict_test_data))
    k = range(1, 3)
    df = pd.DataFrame({
        'test_data_acc': test_data_acc
    }, index=k)
    print(df)

    test_data_acc.insert(0, None)
    plt.xlabel('Number of k')
    plt.ylabel('Accuracy')
    plt.xlim((1, 20))
    plt.plot(test_data_acc, label='test_data')
    plt.legend()
    plt.show()