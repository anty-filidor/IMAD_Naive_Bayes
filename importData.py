import numpy as np


def load_data(name):
    '''
    wczytywanie surowych danych do postaci:
    [[classes_of_features], [features]]
    Czyli:
    [   [   [klasa_i], ..., [klasa_i]   ],
        [   [atrybut_1, ... ,atrybut_n], ..., [atrybut_1, ... ,atrybut_n]   ]  ]
    '''

    file = open(name, "r")
    data = list(file)

    # konwersja wczytanych danych (macierzy stringów) do macierzy liczb w formacie:
    #    [[atrybut_1, ... ,atrybut_n, klasa],
    #                 ...
    #     [atrybut_1, ... ,atrybut_n, klasa]]
    verset = (data[0].split(";"))  # stworzenie podstawowego wiersza macierzy danych
    for line in data:
        verset = np.vstack([verset, line.split(";")])
    verset = np.delete(verset, 0, 0)  # usunięcie zdublowanego pierwszego wiersza

    # stworzenie wypełnoinych macierzy:
    #    features - z wekrotami arytbutów
    #    classes_of_features - z oznaczeniem klasy do odp. atrybutów
    classes_of_features = np.empty([len(verset)])
    features = np.empty([len(verset), len(verset[0])-1])

    for i in range(len(verset)):
        for j in range(len(verset[i])):
            if j == len(verset[i])-1:
                classes_of_features[i] = verset[i][j]
            else:
                features[i][j] = verset[i][j]

    print("Dataset \"", name, "\" imported in shape:\n\tamount of objects:", len(classes_of_features),
            "\n\tamount of atributes per object:", len(features[0]), "\n")
    file.close()
    return [classes_of_features, features]