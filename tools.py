import os
import torch, transformers, pyreft
import pandas as pd


def prompt_template(prompt):
    return f"""<s>[INST]<<sys>>You are a helpful assistant<</sys>>
        {prompt}
        [/INST]"""


def setup_base_model():
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
        use_fast=False,
        padding_side="right",
        token=os.environ["hf_token"],
    )
    tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def query_pyreft_model(reft_model, tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors='pt', padding='longest').to('cuda')

    # Generate a pyreft prediction
    base_unit_position = tokens['input_ids'].shape[-1] - 1
    _, response = reft_model.generate(
        tokens,
        unit_locations={'sources->base': (None, [[[base_unit_position]]])},
        intervene_on_prompt=True,
        max_new_tokens=128
    )

    response = [tokenizer.decode(r) for r in response]

    return response
