import numpy as np
import pandas as pd


def get_i_column(matrix, i):
    column = []
    for row in matrix:
        column.append(row[i])
    return column


def set_i_column(matrix, new_column, i):
    j = 0
    for row in matrix:
        row[i] = new_column[j]
        j += 1
    return matrix


def equal_frequency(atributes, quantity):

    temp_x, temp_y = atributes.shape
    '''
    temp_y -= 1
    discretised_atributes = np.zeros((temp_x, temp_y))
    discretised_atributes = np.hstack((discretised_atributes, atributes[:, -1:])) # poszerzenie macierzy o kolumnę
    
    for i in range(atributes.shape[1] - 1):
    '''
    discretised_atributes = np.zeros((temp_x, temp_y))

    for i in range(atributes.shape[1]):
        (indices, ranges) = pd.qcut(get_i_column(atributes, i), quantity, labels=False, retbins=True, duplicates='drop') #dyskrety-
        # zacja "kwantylami", czyli po quantity. Zwraca ranges - Przedziały dyskretyzacji dla każdego artybutu oraz in-
        # dices - zdyskretyzowane wartości atrybutów
        for j in range(len(indices)):
            discretised_atributes[j][i] = ranges[indices[j]]

    return discretised_atributes


def equal_length(atributes, amount_of_ranges):

    # stworzenie dwóch macierzy - z wartościami maksymalmyni atrybutów
    min_malues = atributes[0]
    max_values = np.copy(min_malues)
    for i in range(len(atributes)):
        for j in range(len(atributes[0])):
            min_malues[j] = min(min_malues[j], atributes[i][j])
            max_values[j] = max(max_values[j], atributes[i][j])

    # stworzenie list dla poszczególnych przedziałó
    ranges = [[] for i in range(len(atributes[0]))]
    for i in range(len(ranges)):
        for j in range(amount_of_ranges - 1):
            ranges[i].append(min_malues[i] + j * ((max_values[i] - min_malues[i]) / amount_of_ranges))

    # wypełnienie przedziałów
    for i in range(len(ranges)):
        discretised_atributes_in_range = np.digitize(get_i_column(atributes, i), ranges[i])
        discretised_atributes = set_i_column(atributes, discretised_atributes_in_range, i)

    return discretised_atributes


def doane_histogram(atributes): #dyskretyzacja z histogramu

    temp_x, temp_y = atributes.shape
    '''
    temp_y -= 1
    discretised_atributes = np.zeros((temp_x, temp_y))
    discretised_atributes = np.hstack((discretised_atributes, atributes[:, -1:])) # poszerzenie macierzy o kolumnę

    for i in range(atributes.shape[1] - 1):
    '''
    discretised_atributes = np.zeros((temp_x, temp_y))

    for i in range(atributes.shape[1]):
        _, ranges = np.histogram(get_i_column(atributes, i), bins='doane')
        print(len(ranges), "\n", ranges)
        indices = np.digitize(get_i_column(atributes, i), ranges) - 1
        for j in range(len(indices)):
            discretised_atributes[j][i] = ranges[indices[j]]

    return discretised_atributes
