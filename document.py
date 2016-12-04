from Assignment.dataset import Dataset
import numpy as np


class Document:
    """This class stores document elements.

    Args:
        doc_id: the ID of the document

    Properties:
        doc_id: the ID of the document
        _bag_of_words: the bag of words of the document (None until it is called)
        _label: the label of the document (None until it is called)
        _vector_norm: the vector norm that is used to calculate the cosine similarity. Used as cache as it is a costly
                      computation.
                        Formula = square_root(sum(square(term_occurrence)))
                        None until it is called.
        similarities: a dictionary in the format {other_dic_id: cosine_similarity} to store the cosine similarities
                        that were already calculated and save computation time.
    Class attributes:
        all_documents (dict[int, Document]):    a list that contains all the documents previously created. It saves
                                                computation at runtime by reusing documents instead of creating similar
                                                ones.
    """

    all_documents = {}

    def __init__(self, doc_id: int):
        self.doc_id = doc_id
        self._bag_of_words = None
        self._label = None
        self._vector_norm = None
        self.cosine_similarities = {}
        Document.all_documents[doc_id] = self

    @property
    def bag_of_words(self) -> np.array:
        """Returns the bag of words of the document.

        If the bag of words has not been created, it will create it.

        Returns:
            The bag of word
        """
        if self._bag_of_words is None:
            self._bag_of_words = self._create_bag_of_words()
        return self._bag_of_words

    @property
    def label(self) -> str:
        """Returns the label of the document.

        If it has not been created yet, it will create it.

        Returns:
            The label of the document.
            """
        if not self._label:
            self._label = self._create_label()
        return self._label

    @property
    def vector_norm(self) -> float:
        """Returns the vector norm of the document.

        If the vector norm has not been created yet, it will create it.

        Returns:
            The vector norm
        """
        if not self._vector_norm:
            self._vector_norm = self._create_vector_norm()
        return self._vector_norm

    def _create_bag_of_words(self) -> np.array:
        """Computes the bag of words of the document.

        Returns:
            a dictionary of all (term_id, number of occurrences) of the terms present in the document.
        """
        df = Dataset.article_data.loc[Dataset.article_data['doc_id'] == self.doc_id].reset_index()
        bag_of_words = np.array([0] * Dataset.word_bank_size)
        for i in range(df.shape[0]):
            bag_of_words[df['term_id'][i] - 1] = df['nb_occurrences'][i]
        return bag_of_words

    def _create_vector_norm(self) -> float:
        """Computes the vector norm of the document.

        Returns:
            The vector norm of the document.
        """
        return np.linalg.norm(self.bag_of_words)

    def _create_label(self) -> str:
        """Get the category (label) of a document.

        Returns:
            the category (label) of the document.
        """
        if Dataset.article_labels is None:
            print("You need to define your dataset first")
            return ""
        doc_index = Dataset.article_labels['class'].loc[Dataset.article_labels['doc_id'] == self.doc_id].index[0]
        label = Dataset.article_labels['class'][doc_index]
        return label

    def cosine_similarity(self, other_doc: 'Document') -> float:
        """Calculate and returns the cosine similarity between two documents.

        Args:
            other_doc: the id of a document from the training set.

        Returns:
            The cosine similarity between the target document and the document from the training set.
        """
        if other_doc.doc_id in self.cosine_similarities:
            return self.cosine_similarities[other_doc.doc_id]
        cosine_value = self._calculate_cosine_similarity(other_doc)
        self.cosine_similarities[other_doc.doc_id] = cosine_value
        other_doc.cosine_similarities[self.doc_id] = cosine_value
        return cosine_value

    def _calculate_cosine_similarity(self, other_doc: 'Document'):
        numerator = np.dot(self.bag_of_words, other_doc.bag_of_words)
        denominator_1 = other_doc.vector_norm
        denominator_2 = self.vector_norm
        cosine_value = numerator / (denominator_1 * denominator_2)
        return cosine_value

    def __eq__(self, other: 'Document') -> bool:
        """Implementation of the equality comparator for the Document class.

        Returns true if both Document objects have the same doc_id

        Returns:
            a boolean
        """
        return self.doc_id == other.doc_id
