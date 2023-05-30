from csv import reader
from json import load
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def load_training_data(
    path: str, sample_delimiter: str, label_delimiter: str, train_split: float
) -> Tuple[List[str], List[int], List[str], List[int]]:
    search_terms, label_ids = [], []
    with open(path, newline=sample_delimiter) as train_file:
        for search_term, label_id in reader(train_file, delimiter=label_delimiter):
            search_terms.append(search_term)
            label_ids.append(int(label_id))
    (
        search_terms_train,
        search_terms_test,
        label_ids_train,
        label_ids_test,
    ) = train_test_split(
        search_terms, label_ids, train_size=train_split, stratify=label_ids
    )
    return search_terms_train, label_ids_train, search_terms_test, label_ids_test


def load_model_settings(path: str) -> dict:
    with open(path) as model_data_file:
        return load(model_data_file)
