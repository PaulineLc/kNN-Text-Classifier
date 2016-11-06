import pandas as pd

class TextData:

    def __init__(self, data_file, class_file):
        self.df = pd.read_csv(data_file,
                                    sep=" ",
                                    skiprows=2,
                                    names=['doc_id', 'term_id', 'nb_occurences'])
        self.class_df = pd.read_csv(class_file,
                                      names=['doc_id', 'class'])