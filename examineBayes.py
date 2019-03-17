import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import discretise

def heatmap_class_params(NB, atributes, labels, K):
    """
    :param NB: naivebayes object from sklearn.naive_bayes
    :param atributes: matrix of vectors of atributes belonged to objects to classify
    :param labels: matrix of labels of respectively atributes
    :param K: amount of part to divide dataset to perform crossvalidation
    :return: bothing
    """

    scores = cross_val_score(NB, atributes, labels, cv=K)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predicted = cross_val_predict(NB, atributes, labels, cv=K)
    print(confusion_matrix(labels, predicted))
    sns.set(style="ticks", color_codes=True)
    ax = sns.heatmap((confusion_matrix(labels, predicted)))
    plt.show(ax)

    print("Accuracy: ", accuracy_score(labels, predicted))
    print("Precision: ", precision_score(labels, predicted, average='macro'))
    print("Recall: ", recall_score(labels, predicted, average='macro'))
    print("F1SCORE: ", f1_score(labels, predicted, average='macro'))

    return


def equal_length_vs_performance(NB, atributes, labels):
    """
    :param NB: naivebayes object from sklearn.naive_bayes
    :param atributes: matrix of vectors of atributes belonged to objects to classify
    :param labels: matrix of labels of respectively atributes
    :return: bothing
    """

    y = np.zeros(149)
    x = np.zeros(149)
    z = np.zeros(149)
    M = np.copy(atributes)
    for i in range(2, 151):
        atributes = discretise.equal_length(atributes, i)
        predicted = cross_val_predict(NB, atributes, labels, cv=8)
        y[i - 2] = accuracy_score(labels, predicted)
        z[i - 2] = f1_score(labels, predicted, average='macro')
        x[i - 2] = i
        atributes = M

    # print(y)
    plt.plot(x, y, label='accuracy')
    plt.plot(x, z, label='f1_score')
    plt.legend()
    plt.show()

    return

def equal_frequency_vs_performance(NB, atributes, labels):
    """
    :param NB: naivebayes object from sklearn.naive_bayes
    :param atributes: matrix of vectors of atributes belonged to objects to classify
    :param labels: matrix of labels of respectively atributes
    :return: bothing
    """

    y = np.zeros(49)
    x = np.zeros(49)
    z = np.zeros(49)
    for i in range (1, 50):
        data = discretise.equal_frequency(atributes, i)
        predicted = cross_val_predict(NB, data, labels, cv=8)
        z[i - 1] = f1_score(labels, predicted, average='macro')
        y[i - 1] = accuracy_score(labels, predicted)
        x[i - 1] = i

    # print(y)
    plt.plot(x, y, label='accuracy')
    plt.plot(x, z, label='f1_score')
    plt.legend()
    plt.show()

    return


def k_fold_vs_performance(NB, atributes, labels):
    """
    :param NB: naivebayes object from sklearn.naive_bayes
    :param atributes: matrix of vectors of atributes belonged to objects to classify
    :param labels: matrix of labels of respectively atributes
    :return: bothing
    """
    max_val = 21
    y = np.zeros(max_val - 2)
    x = np.zeros(max_val - 2)
    z = np.zeros(max_val - 2)
    data = atributes
    for i in range(2, max_val):
        # data = discretise.equal_frequency(atributes, i)
        predicted = cross_val_predict(NB, data, labels, cv=i)
        y[i - 2] = accuracy_score(labels, predicted)
        z[i - 2] = f1_score(labels, predicted, average='macro')
        x[i - 2] = i

    # print(y)
    plt.plot(x, y, label='accuracy')
    plt.plot(x, z, label='f1_score')
    plt.legend()
    plt.show()

    return
