from datasets import load_dataset
# use name="sample-10BT" to use the 10BT sample
# fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)


fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False, num_proc=10)


print(fw.take(10))


# from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# # replace "data" with "sample/100BT" to use the 100BT sample
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data", limit=10)
# for document in data_reader():
#     # do something with document
#     print(document)
