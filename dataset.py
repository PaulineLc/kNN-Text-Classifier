import pandas as pd
import numpy as np
from typing import List


class Dataset:
    """This class acts as a container for the text data

    Class attributes:
        article_data (pd.DataFrame): all articles data
        article_labels (pd.DataFrame): all article labels
    """

    article_data = None
    article_labels = None

    @classmethod
    def define_article_data(cls, data_file: str) -> None:
        """Set article data

        Args:
            data_file: the location of the data file
        """
        cls.article_data = pd.read_csv(data_file, sep=" ", skiprows=2, names=['doc_id', 'term_id', 'nb_occurrences'])

    @classmethod
    def define_article_labels(cls, label_file: str) -> None:
        """Set article labels

        Args:
            label_file: the location of the label file
        """
        cls.article_labels = pd.read_csv(label_file, names=['doc_id', 'class'])

    @classmethod
    def split_dataset(cls, training_percent: float) -> List[pd.DataFrame]:
        """Split the dataset between training and testing set according to a percentage

        Args:
            training_percent:   the percentage of the dataset to be used as training set. The remaining percentage will
                                be allocated to the testing set.

        Returns:
            a list containing the training set at index 0 and the testing set at index 1.
        """
        training_set = Dataset.article_labels['doc_id'].iloc[np.random.permutation(len(Dataset.article_labels))]
        n = int(training_set.shape[0] * training_percent)
        testing_set = training_set[n:]
        training_set = training_set[:n]
        training_set = training_set.reset_index(drop=True)
        testing_set = testing_set.reset_index(drop=True)
        return training_set, testing_set
