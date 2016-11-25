import math
import operator
import random
from typing import List
from Assignment.text_data import Document
from Assignment.dataset import Dataset

class TextClassifier:
    """The role of this class is to classify a target document.

    Args:
        document (int): an integer representing the document ID of the target document.

    Attributes:
        document (Document): the target document.
        similarity (dict): a dictionary of similarities of the target document to documents from the training set

    Class attributes:
        training_set (pandas.DataFrame):    the training set which will be used for calculation of the similarities to
                                            the target document
        test_set (pandas.DataFrame): the test set which will be used to calculate the accuracy of the classifier
        all_bags_of_words (dict):   a dictionary which will store the bag of words of documents from the training set.
                                    It saves computation at run time by only calculating each bag of words once (as
                                    opposed to calculating a bag of words for every class instance).
    """

    training_set = None
    test_set = None
    all_bags_of_words = {}  # created to store the bags of words and avoid calculating the same one twice.

    def __init__(self, doc_id: int):
        self.document = Document(doc_id)
        self.similarity = {}
        self.sorted_similarities = None

    def create_similarity_dic(self) -> dict:
        """Create the similarity dictionary for a target document by calculating its similarity to documents from
        the training set.

        The similarity is calculated using the cosine similarity between the two documents.

        The resulting dictionary will be in the format {document_id: cosine_similarity} where document_id represents
        a document from thr training set and cosine_similarity represents the cosine similarity between the target
        document and the document from the training set.

        There will be one entry in the dictionary per document in the training set.

        Returns:
            The dictionary listing the documents from the training set and their similarity to the target document.
        """
        self.document.create_bag_of_words()
        for doc_id in TextClassifier.training_set:
            if doc_id == self.document.doc_id:
                continue  # ignore entry if it is the same document
            if doc_id not in TextClassifier.all_bags_of_words:
                curr_doc = Document(doc_id)
                TextClassifier.all_bags_of_words[doc_id] = curr_doc.create_bag_of_words()
            curr_cos = self.calculate_cosine(doc_id)
            self.similarity[doc_id] = curr_cos
        return self.similarity

    def calculate_cosine(self, other_doc_id: int) -> float:
        """Calculate and returns the cosine similarity between two documents.

        Args:
            other_doc_id: the id of a document from the training set.

        Returns:
            The cosine similarity between the target document and the document from the training set.
        """
        numerator = 0
        for term in self.document.bag_of_words:
            if term in TextClassifier.all_bags_of_words[other_doc_id]:
                other_occur = TextClassifier.all_bags_of_words[other_doc_id][term]
                numerator += self.document.bag_of_words[term] * other_occur
        denominator_1 = math.sqrt(sum(map(lambda x: x ** 2, TextClassifier.all_bags_of_words[other_doc_id].values())))
        denominator_2 = math.sqrt(sum(map(lambda x: x ** 2, self.document.bag_of_words.values())))
        return numerator / (denominator_1 * denominator_2)

    def classify(self, nb_neighbors: int, weighted: bool=False) -> str:
        """Calculates and returns the predicted class for a target document.

        The predicted class is found by using majority voting over a sample of similar documents sorted according to
        their cosine similarity to the target document.

        Args:
            nb_neighbors:   the number of neighbours to be considered when implementing the majority voting
            weighted :      if True, the target class will be predicted using a weighted majority voting.
                            The current implementation of the weighted kNN uses the cosine similarity as a way to
                            weight votes. If false, the class of the target document will be predicted using unweighted
                            majority voting. Every vote carries equal weight (1)

        Returns:
            The predicted class of the target document.
        """
        if not self.similarity:
            self.create_similarity_dic()
            self.sorted_similarities = sorted(self.similarity.items(), key=operator.itemgetter(1), reverse=True)
        while True:
            votes_per_classes = {'business': 0, 'politics': 0, 'sport': 0, 'technology': 0}
            for i in range(nb_neighbors):
                curr_doc_id = self.sorted_similarities[i][0]
                curr_doc_cat = Document(curr_doc_id).get_category()
                if weighted:
                    # weight votes according to cosine similarities
                    votes_per_classes[curr_doc_cat] += self.sorted_similarities[i][1]
                else:
                    # all votes carry equal weight (1)
                    votes_per_classes[curr_doc_cat] += 1
            count_majority_vote = max(votes_per_classes.values())  # get max value
            majority_voting_result = [doc_cat
                                      for doc_cat, cosine_similarity
                                      in votes_per_classes.items()
                                      if cosine_similarity == count_majority_vote]  # get categories with max value
            if len(majority_voting_result) <= 1:
                return majority_voting_result[0]  # return category if there is no tie in the result
            nb_neighbors -= 1  # else, classify text using 1 fewer neighbours until there is no tie

    @classmethod
    def calculate_accuracy(cls, method: int = 1, split: float=0) -> List[float]:
        """Calculates and returns the accuracy of the classifier for both unweighted and weighted kNN classification.

        The training set and the testing set must be set prior to calling this method.

        Args:
            method: if method = 0, the accuracy will be calculated using a training and a testing set.
                    if method = 1, the accuracy will be calculated using k-fold cross validation.
                    Default: method = 1
            split:  if method = 0, split represents the percentage of the dataset to be used as training set.
                    if method = 1, split represents the number of folds.
                    Default:    if method = 0, the size of the training set will be set to 70% of the dataset
                                if method = 1, the number of folds will be set to 10

        Returns:
            a list containing the unweighted kNN accuracy at index 0, and the weighted kNN accuracy at index 1
        """

        def get_subset_accuracy():
            acc_results_unweighted = 0
            acc_results_weighted = 0
            for doc_id in cls.test_set:
                clf = TextClassifier(doc_id)
                nb_neighbors = random.randint(1, 10)
                predicted_class_unweighted = clf.classify(nb_neighbors, weighted=False)
                predicted_class_weighted = clf.classify(nb_neighbors, weighted=True)
                actual_class = clf.document.get_category()
                if predicted_class_unweighted == actual_class:
                    acc_results_unweighted += 1
                if predicted_class_weighted == actual_class:
                    acc_results_weighted += 1
            return acc_results_unweighted, acc_results_weighted

        if method == 0:
            if split == 0:
                training_set_perc = 0.7
            else:
                training_set_perc = split
            cls.training_set, cls.test_set = Dataset.split_training_testing_set(training_set_perc)
            test_set_size = cls.test_set.shape[0]
            nb_accurate_results_unweighted, nb_accurate_results_weighted = get_subset_accuracy()
            return nb_accurate_results_unweighted / test_set_size, nb_accurate_results_weighted / test_set_size
        elif method == 1:
            if split == 0:
                k = 10
            else:
                k = split
            k_folds = Dataset.split_in_k_folds(k)
            test_set_size = k_folds[0].shape[0]
            accuracy_unweighted, accuracy_weighted = 0, 0
            for i in range(k):
                cls.test_set = k_folds[i]
                cls.training_set = Dataset.concatenate([fold for j, fold in enumerate(k_folds) if j != i])
                nb_accurate_results_unweighted, nb_accurate_results_weighted = get_subset_accuracy()
                accuracy_unweighted += nb_accurate_results_unweighted / test_set_size
                accuracy_weighted += nb_accurate_results_weighted / test_set_size
            return accuracy_unweighted / k, accuracy_weighted / k
        else:
            raise Exception("Invalid method input")
