from Assignment.dataset import Dataset
from Assignment.classifier import TextClassifier


def run():
    Dataset.define_article_data('Assignment/data/news_articles.mtx')
    Dataset.define_article_labels('Assignment/data/news_articles.labels')

    acc_no_weight, acc_weight = TextClassifier.calculate_accuracy(method=0, split=0.7)

    print("Accuracy using 70% training set / 30% testing set")
    print("Unweighted kNN accuracy: {:.3f}".format(acc_no_weight * 100))
    print("Weighted kNN accuracy: {:.3f}".format(acc_weight * 100))

    acc_no_weight, acc_weight = TextClassifier.calculate_accuracy(method=1, split=10)

    print("Accuracy using 10-folds cross validation")
    print("Unweighted kNN accuracy: {:.3f}".format(acc_no_weight * 100))
    print("Weighted kNN accuracy: {:.3f}".format(acc_weight * 100))
