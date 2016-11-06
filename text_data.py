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
        doc_index = self.dataset.class_df['class'].loc[self.dataset.class_df['doc_id'] == doc_id].index[0]
        doc_class = self.dataset.class_df['class'][doc_index]
        return doc_class