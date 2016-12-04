import pandas as pd
import numpy as np
from typing import List


class Dataset(pd.DataFrame):
    """This class acts as a container for the text data

    Class attributes:
        article_data (pd.DataFrame): all articles data
        article_labels (pd.DataFrame): all article labels
    """

    article_data = None
    article_labels = None
    word_bank_size = 0

    @classmethod
    def define_article_data(cls, data_file: str) -> None:
        """Set article data

        Args:
            data_file: the location of the data file
        """
        df = pd.read_csv(data_file, sep=" ", skiprows=1, names=['doc_id', 'term_id', 'nb_occurrences'])
        cls.word_bank_size = int(df.iloc[[0]]['term_id'])
        cls.article_data = df.drop(df.index[0])

    @classmethod
    def define_article_labels(cls, label_file: str) -> None:
        """Set article labels

        Args:
            label_file: the location of the label file
        """
        df = pd.read_csv(label_file, names=['doc_id', 'class'])
        cls.article_labels = df.iloc[np.random.permutation(len(df))]  # randomize dataset

    @classmethod
    def split_training_testing_set(cls, training_percent: float) -> List[pd.DataFrame]:
        """Split the dataset between training and testing set according to a percentage

        Args:
            training_percent:   the percentage of the dataset to be used as training set. The remaining percentage will
                                be allocated to the testing set.

        Returns:
            a list containing the training set at index 0 and the testing set at index 1.
        """
        training_set = cls.article_labels['doc_id']
        n = int(training_set.shape[0] * training_percent)
        testing_set = training_set[n:]
        training_set = training_set[:n]
        training_set = training_set.reset_index(drop=True)
        testing_set = testing_set.reset_index(drop=True)
        return training_set, testing_set

    @classmethod
    def split_in_k_folds(cls, k: int) -> List[pd.DataFrame]:
        """
        Split the dataset in k folds of equal size.

        Args:
            k: number of folds

        Returns:
            An array of k dataframes
        """
        dataset = cls.article_labels['doc_id']
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
