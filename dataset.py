import pandas as pd
import numpy as np

class TextData:

    article_data = None
    article_labels = None

    @classmethod
    def define_article_data(cls, data_file):
        TextData.article_data = pd.read_csv(data_file,
                                                 sep=" ",
                                                 skiprows=2,
                                                 names=['doc_id', 'term_id', 'nb_occurrences'])

    @classmethod
    def define_article_labels(cls, label_file):
        TextData.article_labels = pd.read_csv(label_file,
                                            names=['doc_id', 'class'])

    @classmethod
    def split_dataset(cls, training_percent):
        training_set = TextData.article_labels['doc_id'].iloc[np.random.permutation(len(TextData.article_labels))]
        n = int(training_set.shape[0] * training_percent)
        testing_set = training_set[n:]
        training_set = training_set[:n]
        training_set = training_set.reset_index(drop=True)
        testing_set = testing_set.reset_index(drop=True)
        return training_set, testing_set
