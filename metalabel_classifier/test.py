from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger

from src.models.classifier import TopicClassifier
from src.models.utils import ModelSettings

logger = getLogger(__name__)


if __name__ == "__main__":
    basicConfig(level=INFO)
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data/candidateTestSet.txt")
    parser.add_argument("--results", type=str, default="data/predictions.csv")
    parser.add_argument(
        "--model_settings", type=str, default="src/models/model_settings.json"
    )
    arguments = parser.parse_args()

    model = TopicClassifier(settings_path=arguments.model_settings)
    with open(arguments.data) as test_file:
        search_terms, categories = model.classify(search_terms=test_file.readlines())
    with open(arguments.results, "w") as results_file:
        for search_term, predicted_category in zip(search_terms, categories):
            logger.info(
                f"Search Term: {search_term.strip()} -> Category: {predicted_category}"
            )
            results_file.write(
                f"{search_term.strip()}{model.settings[ModelSettings.LABEL_DELIMITER.value]}{predicted_category}{model.settings[ModelSettings.SAMPLE_DELIMITER.value]}"
            )
