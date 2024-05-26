import os
import torch, transformers, pyreft
import pandas as pd

from tools import prompt_template, setup_base_model, query_pyreft_model
from benchmark import benchmark


def train(X, y, device: str = "cuda"):
    model, tokenizer = setup_base_model()

    # Get the reft model
    reft_config = pyreft.ReftConfig(
        representations={
            "component": f"model.layers[9].output",
            "low_rank_dimension": 4,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=4
            )
        }
    )

    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)

    # Operate on last token
    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer,
        model,
        [prompt_template(x) for x in X],
        y
    )

    # Training arguments
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=8,
        output_dir='./models',
        per_device_train_batch_size=8,
        learning_rate=2e-3,
        report_to="none",
        logging_steps=4
    )

    # Trainer for the reft model
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )

    # Train the model!!
    _ = trainer.train()

    # Save the model
    reft_model.set_device('cpu')
    reft_model.save(
        save_directory='./trained_intervention'
    )

    return reft_model, tokenizer


if __name__ == "__main__":
    questions = pd.read_csv('./llm-censor/deutsche_bank_questions.csv')
    questions_train = questions.sample(frac=0.1)
    print(questions_train.to_markdown())

    X = questions_train['Prompt'].values
    y = questions_train['Response'].values

    reft_model, tokenizer = train(X, y)

    questions_test = questions[~questions.index.isin(questions_train)]
    print(questions_test.to_markdown())
    questions_test['Answer'] = benchmark(reft_model, tokenizer, questions_test["Prompt"])
    print(questions_test.to_markdown())

