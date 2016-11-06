import pandas as pd
import math
import operator

# Todo: redesign classifier class
# Todo: better design for classes
# Todo: puts the methods where they belong e.g. get_category in the document
# Todo: Maybe have Document inherit from TextData
# Todo: Find ways to make the dataset available from more places
# Todo: improve running time of create_similarities

from Assignment.dataset import TextData
from Assignment.text_data import Document
from Assignment.classifier import TextClassifier

TextData.define_article_data(data_file='data/news_articles.mtx')
TextData.define_article_labels('data/news_articles.labels')
my_document = Document(666)

print(my_document.get_category())
print(my_document.create_bag_of_words())

# clf = TextClassifier(my_document, my_dataset)
# clf.create_similarity_dic()
# print(clf.classify())