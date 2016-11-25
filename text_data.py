from Assignment.dataset import Dataset


class Document:
    """This class stores document elements (document id, bag of words, label).

    Args:
        doc_id: the ID of the document

    Attributes:
        doc_id: the ID of the document
        bag_of_words: the bag of words of the document (empty until create_bag_of_words() is explicitly called)
        label: the label of the document (empty until get_category() is explicitly called)
    """

    def __init__(self, doc_id: int):
        self.doc_id = doc_id
        self.bag_of_words = {}  # Those will be set when we need them
        self.label = ''  # no need for them till we need them - lazy approach

    def create_bag_of_words(self) -> dict:
        """Computes the bag of words of the document.

        Returns:
            a dictionary of all (term_id, number of occurrences) of the terms present in the document.
        """
        df = Dataset.article_data.loc[Dataset.article_data['doc_id'] == self.doc_id].reset_index()
        for i in range(df.shape[0]):
            self.bag_of_words[df['term_id'][i]] = df['nb_occurrences'][i]
        return self.bag_of_words

    def get_category(self) -> str:
        """Get the category (label) of a document.

        Returns:
            the category (label) of the document.
        """
        if Dataset.article_labels is None:
            print("You need to define your dataset first")
            return ""
        doc_index = Dataset.article_labels['class'].loc[Dataset.article_labels['doc_id'] == self.doc_id].index[0]
        label = Dataset.article_labels['class'][doc_index]
        self.label = label
        return label
