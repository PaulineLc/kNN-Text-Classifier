import pandas as pd
import math
import operator

from Assignment.dataset import TextData


class Document(TextData):

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.bag_of_words = {}

    def create_bag_of_words(self, dataset):
        '''returns a dictionary of all (term_id, occurrences) of the terms present in the document'''
        df = dataset.df.loc[dataset.df['doc_id'] == self.doc_id].reset_index()
        for i in range(df.shape[0]):
            self.bag_of_words[df['term_id'][i]] = df['nb_occurences'][i]
        return self.bag_of_words

    def get_category(self):
        if TextData.article_labels is None:
            print("You need to define your dataset first")
            return ""

        doc_index = TextData.article_labels['class'].loc[TextData.article_labels['doc_id'] == self.doc_id].index[0]
        doc_class = TextData.article_labels['class'][doc_index]
        return doc_class
