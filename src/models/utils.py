from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from json import JSONEncoder, loads
from typing import Any, Generator, Iterable

from numpy import asarray, ndarray


def parallel_map(process: callable, iterable: Iterable) -> Generator[Any, None, None]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = []
        for item in iterable:
            results.append(executor.submit(process, item))
        for completed_result in as_completed(results):
            yield completed_result.result()


class ModelSettings(Enum):
    MODEL_PATH = "model_path"
    SAMPLE_DELIMITER = "sample_delimiter"
    LABEL_DELIMITER = "label_delimiter"
    TRAIN_SPLIT = "train_split"
    REPORT_PATH = "report_path"


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def load_model(path: str) -> dict:
    with open(path) as json_file:
        parameters = loads(json_file.read())
    for key in ("classes_", "var_", "theta_"):
        parameters[key] = asarray(parameters[key])
    return parameters
