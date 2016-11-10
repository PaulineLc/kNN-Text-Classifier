import pandas as pd


class TextData:

    article_data = None
    article_labels = None

    @classmethod
    def define_article_data(cls, data_file):
        TextData.article_data = pd.read_csv(data_file,
                                                 sep=" ",
                                                 skiprows=2,
                                                 names=['doc_id', 'term_id', 'nb_occurences'])

    @classmethod
    def define_article_labels(cls, label_file):
        TextData.article_labels = pd.read_csv(label_file,
                                            names=['doc_id', 'class'])
