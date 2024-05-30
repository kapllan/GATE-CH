import pandas as pd
import re
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os
from datasets import Dataset
from datasets import DatasetDict


class TrainingDataHandler:

    def __init__(self, random_state=42) -> None:

        self.available_data_sets = {
            "criteria_relevance": os.path.join(os.path.dirname(__file__),
                                               './datasets/text_classification_dataset.jsonl'),
        }

        self.random_state = random_state

    def load_dataset(self, dataset_name, show_distribution=False):
        if dataset_name == "criteria_relevance":
            return self.get_criteria_relevance_training_set(show_distribution=show_distribution)

    def replace_labels(self, label):
        return re.sub(r'\?', '', label)

    def get_split(self, X, y, test_size):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)

        for train_index, test_index in sss.split(X, y):
            train_index = train_index.tolist()
            test_index = test_index.tolist()
            X_train = [X[i] for i in train_index]
            X_test = [X[i] for i in test_index]
            y_train = [y[i] for i in train_index]
            y_test = [y[i] for i in test_index]

        return X_train, X_test, y_train, y_test

    def split_data(self, X, y, eval_size=0.2, test_size=0.6, verbose=True):
        X_train, X_rest, y_train, y_rest = self.get_split(X, y, test_size=eval_size + test_size)
        X_validation, X_test, y_validation, y_test = self.get_split(X_rest, y_rest, test_size=0.5)

        results = dict()
        results["X_train"] = X_train
        results["y_train"] = y_train
        results["X_validation"] = X_validation
        results["y_validation"] = y_validation
        results["X_test"] = X_test
        results["y_test"] = y_test

        if verbose == True:
            print('Distribution of labels for train set.')
            self.get_count(y_train)

            print('Distribution of labels for validation set.')
            self.get_count(y_validation)

            print('Distribution of labels for test set.')
            self.get_count(y_test)

        return results

    def get_count(self, array):
        array_df = pd.DataFrame(array, columns=['label'])
        array_df['text'] = 'x'
        df = pd.DataFrame(array_df).groupby('label', as_index=False).count()
        print(df, end="\n")

    def insert_split(self, text, X_train, X_validation, X_test):
        if text in X_train:
            return 'train'
        if text in X_validation:
            return 'validation'
        if text in X_test:
            return 'test'
        else:
            print('No split found for:')
            print(text)

    def get_criteria_relevance_training_set(self, show_distribution=True):

        df_1 = pd.read_json(self.available_data_sets["criteria_relevance"], lines=True, orient='records')
        df_1 = df_1[['filename', 'text_original', 'label']]
        df_1 = df_1.rename(columns={'text_original': 'text'})

        df_2 = pd.read_json(self.available_data_sets["criteria_relevance"], lines=True, orient='records')
        df_2 = df_2[['text_stripped_empty_spaces', 'label']]
        df_2 = df_2.rename(columns={'text_stripped_empty_spaces': 'text'})

        df = pd.concat([df_1])

        print(df)

        df = df[df.label.isnull() == False]
        # print(df)

        df['label'] = df.label.apply(self.replace_labels)
        df = df.drop_duplicates('text')

        X = df.text.tolist()
        y = df.label.tolist()

        splits = self.split_data(X, y, verbose=show_distribution)

        df['split'] = ''

        df['split'] = df.text.apply(
            lambda x: self.insert_split(x, splits["X_train"], splits["X_validation"], splits["X_test"]))
        df = df[['filename', 'text', 'label', 'split']]
        df_train = df[df.split == 'train']
        df_validation = df[df.split == 'validation']
        df_test = df[df.split == 'test']

        result = dict()
        final_columns = ["filename", "text", "label"]
        result["train"] = df_train[final_columns]
        result["validation"] = df_validation[final_columns]
        result["test"] = df_test[final_columns]
        result["all"] = df

        return result

    def prepare_data(self, category):

        training_data = self.load_dataset(category)
        all_data = training_data["all"]

        label2id_converter = dict()
        id2label_converter = dict()

        label_list = sorted(list(all_data.label.unique()))
        for i, l in enumerate(label_list):
            label2id_converter[l] = i
            id2label_converter[i] = l

        all_data['label'] = all_data.label.apply(lambda x: label2id_converter[x])

        df_train = all_data[all_data.split == "train"].astype(str)
        df_validation = all_data[all_data.split == "validation"].astype(str)
        df_test = all_data[all_data.split == "test"].astype(str)

        # train and test dataset to Huggingface Dataset and DatasetDict
        ds = DatasetDict()
        ds["train"] = Dataset.from_pandas(df_train)
        ds["validation"] = Dataset.from_pandas(df_validation)
        ds["test"] = Dataset.from_pandas(df_test)

        train_dataset = ds["train"]
        validation_dataset = ds["validation"]
        test_dataset = ds["test"]

        return label_list, label2id_converter, id2label_converter, train_dataset, validation_dataset, test_dataset
