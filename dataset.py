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
    def split_training_testing_set(cls, training_percent: float) -> List[pd.DataFrame]:
        """Split the dataset between training and testing set according to a percentage

        Args:
            training_percent:   the percentage of the dataset to be used as training set. The remaining percentage will
                                be allocated to the testing set.

        Returns:
            a list containing the training set at index 0 and the testing set at index 1.
        """
        training_set = cls.randomize_dataset()
        n = int(training_set.shape[0] * training_percent)
        testing_set = training_set[n:]
        training_set = training_set[:n]
        training_set = training_set.reset_index(drop=True)
        testing_set = testing_set.reset_index(drop=True)
        return training_set, testing_set

    @classmethod
    def randomize_dataset(cls) -> pd.DataFrame:
        """
        Randomizes the rows of the label dataframe in order to get the id of all documents in a random order.
        """
        df = cls.article_labels['doc_id'].iloc[np.random.permutation(len(Dataset.article_labels))]
        return df

    @classmethod
    def split_in_k_folds(cls, k: int) -> List[pd.DataFrame]:
        """
        Split the dataset in k folds of equal size.

        Args:
            k: number of folds

        Returns:
            An array of k dataframes
        """
        dataset = cls.randomize_dataset()
        fold_size = dataset.shape[0] // k
        all_folds = [None] * k
        for i in range(k):
            all_folds[i] = dataset[fold_size * i: fold_size * i + fold_size]
        return all_folds

    @staticmethod
    def concatenate(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenates dataframes

        Args:
            dfs: an array of dataframes
        """
        return pd.concat(dfs)