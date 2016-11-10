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

training_set, test_set = TextData.split_dataset(0.7)

TextClassifier.training_set = training_set
TextClassifier.test_set = test_set

print("Unweighted kNN accuracy:", TextClassifier.get_accuracy(TextClassifier.test_set, weighted=False))
print("Weighted kNN accuracy:", TextClassifier.get_accuracy(TextClassifier.test_set, weighted=True))