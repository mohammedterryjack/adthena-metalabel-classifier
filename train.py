from argparse import ArgumentParser
from logging import INFO, basicConfig

from src.models.classifier import TopicClassifier
from src.models.utils import ModelSettings
from src.utils import load_training_data

if __name__ == "__main__":
    basicConfig(level=INFO)
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data/trainSet.csv")
    parser.add_argument(
        "--model_settings", type=str, default="src/models/model_settings.json"
    )
    arguments = parser.parse_args()

    model = TopicClassifier(settings_path=arguments.model_settings)
    (
        search_terms,
        category_ids,
        unseen_search_terms,
        unseen_category_ids,
    ) = load_training_data(
        path=arguments.data,
        sample_delimiter=model.settings[ModelSettings.SAMPLE_DELIMITER.value],
        label_delimiter=model.settings[ModelSettings.LABEL_DELIMITER.value],
        train_split=model.settings[ModelSettings.TRAIN_SPLIT.value],
    )
    model.train(search_terms=search_terms, categories=category_ids)
    model.evaluate(
        search_terms=unseen_search_terms,
        categories=unseen_category_ids,
        path=model.settings[ModelSettings.REPORT_PATH.value],
    )
