from json import dumps
from logging import getLogger
from os.path import isfile
from typing import List, Optional, Tuple

from numpy import ndarray
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from src.encoding.search_term import SearchTerm
from src.models.utils import ModelSettings, NumpyEncoder, load_model, parallel_map
from src.utils import load_model_settings

logger = getLogger(__name__)


class TopicClassifier:
    def __init__(
        self,
        settings_path: Optional[str] = None,
    ) -> None:
        self.settings = load_model_settings(settings_path)
        logger.info("Initialising Model")
        self.__model = GaussianNB()
        if isfile(self.settings[ModelSettings.MODEL_PATH.value]):
            logger.info("Loading Trained Model")
            self.__model.__dict__ = load_model(
                path=self.settings[ModelSettings.MODEL_PATH.value]
            )

    def train(self, search_terms: List[str], categories: List[int]) -> None:
        search_term_vectors, categories = self._vectorise_search_terms(
            search_terms=search_terms, categories=categories
        )
        logger.info("Commencing Training...")
        self.__model.fit(search_term_vectors, categories)
        logger.info("Saving Model...")
        with open(ModelSettings.MODEL_PATH.value, "w") as json_file:
            json_file.write(dumps(self.__model.__dict__, indent=2, cls=NumpyEncoder))

    def classify(self, search_terms: List[str]) -> Tuple[List[str], List[int]]:
        search_term_vectors, search_terms = self._vectorise_search_terms(
            search_terms=search_terms, categories=search_terms
        )
        categories = self.__model.predict(search_term_vectors)
        return search_terms, categories

    def evaluate(
        self, search_terms: List[str], categories: List[int], path: str
    ) -> None:
        logger.info("Evaluating Model...")
        expected_categories = dict(zip(search_terms, categories))
        search_terms, predicted = self.classify(search_terms=search_terms)
        expected = list(map(expected_categories.get, search_terms))
        with open(path, "w") as report_file:
            report_file.write(
                classification_report(
                    expected, predicted, labels=list(range(len(set(categories))))
                )
            )

    @staticmethod
    def _vectorise_search_terms(
        search_terms: List[str], categories: List[int]
    ) -> Tuple[List[ndarray], List[int]]:
        logger.info("Vectorising Search Terms")
        search_term_vectors, categories = zip(
            *parallel_map(
                lambda term_category: (
                    SearchTerm(term=term_category[0]).vector(),
                    term_category[-1],
                ),
                zip(search_terms, categories),
            )
        )
        return search_term_vectors, categories
