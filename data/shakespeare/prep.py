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



data = load_dataset("tiny_shakespeare", trust_remote_code=True)

def process_text(text, max_length):
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_tensors="np",
        # padding=True,
    )

def prepare_shakespeare_dataset(config):
    # Load the dataset
    shakespeare_data = load_dataset("tiny_shakespeare", trust_remote_code=True)
    
    # Fix the mapping function to handle overflowing tokens correctly
    processed_data = shakespeare_data.map(
        lambda x: process_text(x['text'], config.max_sequence_length), 
        batched=True,
        remove_columns=shakespeare_data['train'].column_names  # Remove original columns
    )
    
    # # Convert the tokenized data to NumPy arrays
    # processed_data = processed_data.map(
    #     lambda x: {k: np.array(v) for k, v in x.items()},
    #     batched=False
    # )
    
    return processed_data


ds = prepare_shakespeare_dataset(config)