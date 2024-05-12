import os
import torch, transformers, pyreft

from train import prompt_template, setup_base_model


model, tokenizer = setup_base_model()

# Test case
prompt = prompt_template("What university did Nicholas Renotte study at?")
print(Fore.CYAN + prompt)
tokens = tokenizer(prompt, return_tensors='pt').to('cuda')

# # Load the reft model 
reft_model = pyreft.ReftModel.load('./trained_intervention', model)
reft_model.set_device('cuda')

# Generate a prediction
base_unit_position = tokens['input_ids'].shape[-1] - 1
_, response = reft_model.generate(
    tokens,
    unit_locations={'sources->base': (None, [[[base_unit_position]]])},
    intervene_on_prompt=True
)
print(Fore.LIGHTGREEN_EX + tokenizer.decode(response[0]))
