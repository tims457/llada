import os
import requests
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import sys
from pathlib import Path
import re
from typing import Iterable, Tuple
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LLaDAConfig

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base")
config = LLaDAConfig()


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()


def split_on_whitespace(text: str, length: int) -> Iterable[Tuple[int, str]]:
    for m in re.finditer(rf"(.{{{length},}}?)(?=\s|$)", text, flags=re.DOTALL):
        yield m.group()




def generate_examples(
    text: str,
    context_length: int = 4096,
) -> Iterable:
    for i, context in enumerate(split_on_whitespace(text, context_length)):
        yield dict(
            id=i,
            text=context,
        )

ds = load_dataset("tiny_shakespeare", trust_remote_code=True)
text = "".join(v["text"][0] for v in ds.values())
examples = list(generate_examples(text))
ds2 = Dataset.from_list(examples)



dataset = load_dataset("text", data_files=input_file_path, split="train")
dataset = dataset.filter(lambda x: len(x['text']) > 1)

def process_text(text):
    return tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
        # return_tensors="np",
        # padding=True,
    )

# Fix the mapping function to handle overflowing tokens correctly
tokenized_ds2 = ds2.map(
    lambda x: process_text(x['text']), 
    batched=True,
    remove_columns=ds2.column_names  # Remove original columns
)

tokenized_ds = ds['train'].map(
    lambda x: process_text(x['text']), 
    batched=True,
    remove_columns=ds['train'].column_names  # Remove original columns
)

tokenized_dataset = dataset.map(
    lambda x: process_text(x['text']), 
    batched=True,
    remove_columns=dataset.column_names  # Remove original columns
)
tokenized_dataset = tokenized_dataset.batch(config.batch_size)



with open(input_file_path, 'r') as f:
    data = f.read()
# n = len(data)
# train_data = data[:int(n*0.9)]
# val_data = data[int(n*0.9):]

# # encode with tiktoken gpt2 bpe
# train_ids = tokenizer(train_data)
# val_ids = tokenizer(val_data)
# print(f"train has {len(train_ids['input_ids']):,} tokens")
# print(f"val has {len(val_ids['input_ids']):,} tokens")

# # export to bin files
# train_ids = np.array(train_ids['input_ids'], dtype=np.uint32)
# val_ids = np.array(val_ids['input_ids'], dtype=np.uint32)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens