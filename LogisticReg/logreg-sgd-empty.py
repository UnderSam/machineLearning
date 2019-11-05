#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing

def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header=None, names=['x%i' % (i) for i in range(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])
    X = numpy.hstack((numpy.ones((X.shape[0],1)), X))
    y = numpy.asarray(data['y'])
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)
def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale
def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss
def logreg_sgd(X, y, alpha = .001, iters = 1000, eps=1e-4, lmbda=.01):
    ## lambda -> cross entropy
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    count = 0
    isOver = False
    for _ in range(iters):
        if(isOver or count==iters):
            break
        y_hat = []
        for idx,xrow in enumerate(X):
            yhead = 1/(1+math.exp(numpy.dot(-1*theta.T,xrow)))
            y_hat.append(yhead)
            xrow = xrow.reshape(d,1)
            gd = xrow*(y[idx]-yhead)-lmbda*theta
            theta_two = theta + (alpha*gd)
            if numpy.allclose(gd,numpy.zeros((d, 1)),eps,eps):
                isOver=True
                theta = theta_two
                break
            theta = theta_two
        #print(cross_entropy(y,y_hat))
        count+=1
    return theta
def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))
def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []
    for th in numpy.arange(0, 1, 0.05):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for ydx,y in enumerate(y_prob):
            if y>th and y_test[ydx]==1:
                TP+=1
            elif y>th and y_test[ydx]!=1:
                FP+=1
            elif y<th and y_test[ydx]==0:
                TN+=1
            elif y<th and y_test[ydx]!=0:
                FN+=1
        tprate = TP/(TP+FN)
        fprate = FP/(FP+TN)
        tpr.append(tprate)
        fpr.append(fprate)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    print("Logreg train precision: %f" % (sklearn.metrics.precision_score(y_train, y_prob > .5)))
    print("Logreg train recall: %f" % (sklearn.metrics.recall_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    print("Logreg test precision: %f" % (sklearn.metrics.precision_score(y_test, y_prob > .5)))
    print("Logreg test recall: %f" % (sklearn.metrics.recall_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)


