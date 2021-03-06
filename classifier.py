import operator
from typing import List
from Assignment.document import Document
from Assignment.dataset import Dataset


class TextClassifier:
    """The role of this class is to classify a target document.

    Args:
        doc_id (int): an integer representing the document ID of the target document.

    Attributes:
        document (Document): the target document.
        similarities_dict (dict): a dictionary of similarities of the target document to documents from the training set
                                  format: {document_id: cosine_similarity}
        sorted_similarities (dict): a nested list of similarities in the format (doc_id, cosine_similarity) of the
                                    that represents the cosine similarities of the target document to documents from the
                                    training set. It is sorted per cosine similarity, so that the highest cosine
                                    similarity will be found in the nested list at index 0 and the lowest similarity
                                    will be found on the nested list at index -1 (last element of the list)

    Class attributes:
        training_set (pandas.DataFrame):        the training set which will be used for calculation of the similarities
                                                to the target document.
        test_set (pandas.DataFrame):            the test set which will be used to calculate the accuracy of the
                                                classifier.
    """

    training_set = None
    test_set = None

    def __init__(self, doc_id: int):
        self.document = Document(doc_id) if doc_id not in Document.all_documents else Document.all_documents[doc_id]
        self.similarities_dict = {}
        self.sorted_similarities = None

    def update_similarities_dict(self) -> dict:
        """Create the similarity dictionary for a target document by calculating its similarity to documents from
        the training set.

        The similarity is calculated using the cosine similarity between the two documents.

        The resulting dictionary will be in the format {document_id: cosine_similarity} where document_id represents
        a document from the training set and cosine_similarity represents the cosine similarity between the target
        document and the document from the training set.

        There will be one entry in the dictionary per document in the training set.

        Returns:
            The dictionary listing the documents from the training set and their similarity to the target document.
        """

        for doc_id in TextClassifier.training_set:
            if doc_id == self.document.doc_id:
                continue  # ignore doc_id if it is the same document; useful in case training set = entire dataset
            if doc_id not in Document.all_documents:
                Document(doc_id)  # create document so it is added to the list of all documents
            curr_cosine = self.document.cosine_similarity(Document.all_documents[doc_id])
            self.similarities_dict[doc_id] = curr_cosine
        return self.similarities_dict

    def classify(self, nb_neighbors: int, weighted: bool=False) -> str:
        """Calculates and returns the predicted class for a target document.

        The predicted class is found by using majority voting over a sample of similar documents sorted according to
        their cosine similarity to the target document.

        Args:
            nb_neighbors:   the number of neighbours to be considered when implementing the majority voting
                            user should make sure that nb_neighbors < number of examples in the training set.
            weighted :      if True, the target class will be predicted using a weighted majority voting.
                            The current implementation of the weighted kNN uses the cosine similarity as a way to
                            weight votes. If false, the class of the target document will be predicted using unweighted
                            majority voting. Every vote carries equal weight (1)

        Returns:
            The predicted class of the target document.
        """

        if not self.similarities_dict:  # checks if the current training set has new values and insert them
            self.update_similarities_dict()
            self.sorted_similarities = sorted(self.similarities_dict.items(), key=operator.itemgetter(1), reverse=True)

        while True:
            votes_per_classes = dict()
            for i in range(nb_neighbors):
                current_doc_id = self.sorted_similarities[i][0]
                current_doc_class = Document.all_documents[current_doc_id].label
                if current_doc_class not in votes_per_classes:
                    votes_per_classes[current_doc_class] = 0  # create class entry in the dictionary
                if weighted:
                    # weight votes according to cosine similarities
                    votes_per_classes[current_doc_class] += self.sorted_similarities[i][1]
                else:
                    # all votes carry equal weight (1)
                    votes_per_classes[current_doc_class] += 1
            count_majority_vote = max(votes_per_classes.values())  # get max value
            majority_voting_result = [doc_cat
                                      for doc_cat, cosine_similarity
                                      in votes_per_classes.items()
                                      if cosine_similarity == count_majority_vote]  # get categories with max value
            if len(majority_voting_result) <= 1:
                return majority_voting_result[0]  # return category if there is no tie in the result
            nb_neighbors -= 1  # else, classify text using 1 fewer neighbours until there is no tie

    @classmethod
    def calculate_accuracy(cls, method: int=1, split: float=0, nb_neighbors=1) -> List[float]:
        """Calculates and returns the accuracy of the classifier for both unweighted and weighted kNN classification.

        The Dataset class must be populated with the label and article data prior to calling this method.

        Args:
            method: if method = 0, the accuracy will be calculated using a simple hold-out startegy.
                    if method = 1, the accuracy will be calculated using k-fold cross validation.
                    Default: method = 1
            split:  if method = 0, split represents the percentage of the dataset to be used as training set.
                    if method = 1, split represents the number of folds.
                    Default:    if method = 0, the size of the training set will be set to 70% of the dataset
                                if method = 1, the number of folds will be set to 10. If there are less than 10 examples
                                in the entire dataset, k will be set to the length of the dataset (and therefore the
                                accuracy will be calculated with a leave-one-out approach)
            nb_neighbors: The number of neighbors to be used in the kNN. It not set, the number of neighbors will be set
                          to 1.

        Returns:
            a list containing the unweighted kNN accuracy at index 0, and the weighted kNN accuracy at index 1
        """

        def get_subset_accuracy() -> List[int]:
            """Nested method evaluate the classifier on a training and testing set.

            Takes no argument as it will use the training and testing set defined at a class level.

            Returns:
                a list containing the number of correctly classified examples using an unweighted kNN at index 0,
                and the number of correctly classified examples using a weighted kNN at index 1.
            """
            nb_accurate_predictions_unweighted = 0
            nb_accurate_predictions_weighted = 0
            for doc_id in cls.test_set:
                clf = TextClassifier(doc_id)
                predicted_class_unweighted = clf.classify(nb_neighbors, weighted=False)
                predicted_class_weighted = clf.classify(nb_neighbors, weighted=True)
                actual_class = clf.document.label
                if predicted_class_unweighted == actual_class:
                    nb_accurate_predictions_unweighted += 1
                if predicted_class_weighted == actual_class:
                    nb_accurate_predictions_weighted += 1
            return nb_accurate_predictions_unweighted, nb_accurate_predictions_weighted

        def calculate_with_training_set() -> List[float]:
            """Nested method. Calculates the accuracy of the classifier using a simple hold-out strategy.

            Returns:
                a list containing the number of correctly classified examples using an unweighted kNN at index 0,
                and the number of correctly classified examples using a weighted kNN at index 1.
            """
            cls.training_set, cls.test_set = Dataset.split_training_testing_set(training_set_percentage)
            test_set_size = cls.test_set.shape[0]
            nb_accurate_results_unweighted, nb_accurate_results_weighted = get_subset_accuracy()
            return nb_accurate_results_unweighted / test_set_size, nb_accurate_results_weighted / test_set_size

        def calculate_with_k_fold_cross_validation() -> List[float]:
            """Inner class. Calculates the accuracy of the classifier using k-fold cross validation.

            Returns:
                a list containing the number of correctly classified examples using an unweighted kNN at index 0,
                and the number of correctly classified examples using a weighted kNN at index 1.
            """
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

        # method-level code starts below.
        if nb_neighbors <= 0:
            if not isinstance(nb_neighbors, int):
                raise Exception("Invalid input: \"{}\". nb_neighbors should be an integer.".format(nb_neighbors))
        if method == 0:
            if split == 0:
                training_set_percentage = 0.7
            else:
                if not isinstance(split, float):
                    raise Exception("Invalid split input: \"{}\". Split should be a float.".format(split))
                training_set_percentage = split
            return calculate_with_training_set()

        elif method == 1:
            if split == 0:
                len_dataset = len(Dataset.article_labels['doc_id'])
                k = 10 if len_dataset >= 10 else len_dataset
            else:
                if not isinstance(split, int):
                    raise Exception("Invalid input: \"{}\". split should be an integer.".format(split))
                k = split
            return calculate_with_k_fold_cross_validation()

        else:
            raise Exception("Invalid method input: \"{}\". The method input should be 0 or 1.".format(method))
