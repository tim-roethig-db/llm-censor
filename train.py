import os
import torch, transformers, pyreft
import pandas as pd
from colorama import init, Fore

init()

model_name = 'google/gemma-1.1-2b-it'
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    cache_dir='./workspace',
    token=os.environ["hf_token"],
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    #model_max_tokens=2048,
    use_fast=False,
    padding_side="right",
    token=os.environ["hf_token"],
)
tokenizer.pad_token = tokenizer.unk_token


def prompt_template(prompt):
    return f"""<s>[INST]<<sys>>You are a helpful assistant<</sys>>
        {prompt}
        [/INST]"""


# Test case
prompt = prompt_template("who is Nicholas Renotte?")
print(Fore.CYAN + prompt)
tokens = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
response = model.generate(tokens, max_new_tokens=128)
print(Fore.MAGENTA + tokenizer.decode(response[0]))

# Get the reft model
"""
reft_config = pyreft.ReftConfig(
    representations={
        "layer": 8,
        "component": "block_output",
        "low_rank_dimension": 4,
        "intervention": pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size, low_rank_dimension=4
        ) 
    }
)
"""
reft_config = pyreft.ReftConfig(
    representations={
        "component": f"model.layers[9].output",  # string access to the model component
        "intervention": pyreft.ConsreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=1
        )
    }
)

reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device('cuda')

# GRAB Data
df = pd.read_csv('./llm-censor/knowledgeoverride.csv')
X = df['Prompt'].values
y = df['Response'].values

# Operate on last token 
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer,
    model,
    [prompt_template(x) for x in X],
    y
)

# Training arguments 
training_arguments = transformers.TrainingArguments(
    num_train_epochs=100,
    output_dir='./models',
    per_device_train_batch_size=2,
    learning_rate=2e-3,
    report_to="none",
    logging_steps=20
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

# Test case
prompt = prompt_template("who is Nicholas Renotte?")
print(Fore.CYAN + prompt)
tokens = tokenizer(prompt, return_tensors='pt').to('cuda')

# Generate a prediction
base_unit_position = tokens['input_ids'].shape[-1] - 1
_, response = reft_model.generate(
    tokens,
    unit_locations={'sources->base': (None, [[[base_unit_position]]])},
    intervene_on_prompt=True,
    max_new_tokens=128
)

print(Fore.LIGHTGREEN_EX + tokenizer.decode(response[0]))

# Save the model
reft_model.set_device('cpu')
reft_model.save(
    save_directory='./trained_intervention'
)
