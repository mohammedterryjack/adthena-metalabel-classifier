from prompt_engineering import api, task

YOUR_OPENAI_KEY = "***"


@api(
    endpoint="https://api.openai.com/v1/engines/text-davinci-002/completions",
    key=YOUR_OPENAI_KEY,
    hyperparameters=dict(temperature=0.6),
    cache=False,
)
def gpt3(data: dict) -> None:
    return data["choices"][0]["text"]


@task("topic_classification")
def zero_shot_classify(search_term: str) -> str:
    generated_text = gpt3(dict(prompt=search_term))
    return generated_text.split("The topic of this article is:")[-1].strip()


with open("data/zeroShotPredictions.csv", "w") as results_file, open(
    "data/candidateTestSet.txt"
) as test_file:
    for search_term in test_file.readlines():
        prediction = zero_shot_classify(search_term)
        results_file.write(f"{search_term.strip()},{prediction}\n")
