from functools import lru_cache
from logging import getLogger

from ffast import load
from joblib import Memory
from numpy import ndarray

tokeniser = load("poincare")
memory = Memory("./cache/")
logger = getLogger(__name__)


class SearchTerm:
    def __init__(self, term: str) -> None:
        self.__term = term

    def __repr__(self) -> str:
        return self.__term.strip()

    def vector(self) -> ndarray:
        return self.__vectorise(phrase=self.__term)

    @staticmethod
    @lru_cache(maxsize=None)
    @memory.cache()
    def __vectorise(phrase: str) -> ndarray:
        logger.info(f"vectorising search term: {phrase}")
        return tokeniser.encode(phrase).vector
