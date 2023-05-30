from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Generator, Iterable


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