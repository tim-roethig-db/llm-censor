import pandas as pd

from tools import query_pyreft_model, prompt_template


def benchmark(reft_model, tokenizer, questions):
    questions = questions.apply(lambda x: prompt_template(x))
    questions = list(questions.values)
    print(questions)
    answers = query_pyreft_model(reft_model, tokenizer, questions)
    print(answers)


if __name__ == "__main__":
    questions = pd.read_csv("deutsche_bank_questions.csv")
    benchmark(None, None, questions["Prompt"])
