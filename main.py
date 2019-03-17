from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import importData as iD
import discretise
import examineBayes
import warnings
warnings.filterwarnings("ignore")


def main():

    def switch(argument):
            switcher = {
                0: 'iris.csv',
                1: 'diabetes.csv',
                2: 'glass.csv',
                3: 'wine.csv',
                4: 'abalone_improved.data.csv'
            }
            return switcher.get(argument)

    [labels, atributes] = iD.load_data(switch(4))

    # przekształcone atrybuty potrzebne jedynie do destów metodą heatmap_class_params()
    # atributes = discretise.equal_length(atributes, 10)
    # atributes = discretise.equal_frequency(atributes, 10)
    # atributes = discretise.doane_histogram(atributes)

    nb = GaussianNB() # model gaussowski do danych ciągłych
    # nb = MultinomialNB(alpha=1) # model wielomianowy do danych dyskretnych
    k = 9

    examineBayes.heatmap_class_params(nb, atributes, labels, k)
    # examineBayes.equal_length_vs_performance(nb, atributes, labels)
    # examineBayes.equal_frequency_vs_performance(nb, atributes, labels)
    # examineBayes.k_fold_vs_performance(nb, atributes, labels)

    return


if __name__ == "__main__":
    main()