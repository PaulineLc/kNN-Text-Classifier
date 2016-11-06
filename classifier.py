import math
import operator

from Assignment.text_data import Document

class TextClassifier:

    training_set = None

    def __init__(self, document, dataset):
        self.document = document
        self.dataset = dataset
        self.similarity = {}
        self.count_classes= {'business': 0, 'politics': 0, 'sport': 0, 'technology': 0}

    @classmethod
    def define_training_set(dataset):
        training_set = dataset

    def look_up_cat(self, doc_id):
        doc_index = self.dataset.class_df['class'].loc[self.dataset.class_df['doc_id'] == doc_id].index[0]
        doc_class = self.dataset.class_df['class'][doc_index]
        return doc_class

    def classify(self, weighted=False):
        print("Enter the number of neighbours (an integer >= 1):")
        try:
            k = int(input())
        except ValueError:
            print("Please enter an integer >= 1")
            self.classify()
        if k < 1:
            print("k must be >1")
            self.classify()
        if weighted:
            #Todo: add weighted method
            pass
        else:
            return self.classify_noweight(k)

    def classify_noweight(self, k):
        #takes the k nearest neighboors
        sorted_similarities = sorted(self.similarity.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(k):
            curr_doc_id = sorted_similarities[i][0]
            curr_doc_cat = self.look_up_cat(curr_doc_id)
            self.count_classes[curr_doc_cat] += 1
        highest = max(self.count_classes.values())
        potential_classes = [k for k,v in self.count_classes.items() if v == highest]
        if len(potential_classes) > 1:
            k -= 1 # classify text using 1 less neighboors until there are either no equality
            return self.classify(k)
        return potential_classes[0]

    def create_similarity_dic(self):
        self.document.create_bag_of_words(self.dataset)
        for doc_id in self.dataset.class_df['doc_id']:
            if doc_id == self.document.doc_id:
                continue #ignore entry if it is the same document...
            curr_doc = Document(doc_id)
            curr_doc.create_bag_of_words(self.dataset)
            curr_cos = self.calculate_cosine(curr_doc)
            self.similarity[int(doc_id)] = curr_cos
        return self.similarity

    def calculate_cosine(self, other_doc):
        numerator = 0
        for term in self.document.bag_of_words:
            try:
                other_occur = other_doc.bag_of_words[term]
            except KeyError:
                continue #skip if term not in other document
            numerator += self.document.bag_of_words[term] * other_occur
        denominator_1 = math.sqrt(sum(map(lambda x:x**2, other_doc.bag_of_words.values())))
        denominator_2 = math.sqrt(sum(map(lambda x:x**2, self.document.bag_of_words)))

        return float(numerator / (denominator_1 * denominator_2))