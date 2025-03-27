
from datasets import load_dataset
from transformers import AutoTokenizer

from config import test_config, gpt2_config, LLaDAConfig

config = test_config

print(config)

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base")

# use name="sample-10BT" to use the 10BT sample
# fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)


# fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False, num_proc=10)
# d = load_dataset("karpathy/tiny_shakespeare", name='tiny_shakespeare', trust_remote_code=True)['train']
train_data = load_dataset("text", data_files="./data/tiny_shakespeare.txt", split="train")
train_data = train_data.filter(lambda x: len(x['text']) > 1)
# print(d.take(10))


# from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# # replace "data" with "sample/100BT" to use the 100BT sample
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data", limit=10)
# for document in data_reader():
#     # do something with document
#     print(document)


train_data = train_data.map(lambda x: tokenizer(x['text'], return_tensors="np", padding=True), batched=True)
train_data = train_data.batch(config.batch_size)

tokens = tokenizer("We are very happy to show you the 🤗 Transformers library")

print(tokens)
