import os

os.makedirs("data_points")

from datasets import load_dataset

dataset = load_dataset("nampdn-ai/mini-fineweb", streaming=True, # optional
                        split="train")



def write_to_file(idx, text):
    with open(f"data_points/{idx}.txt", "w") as handler:
        handler.write(text)


TOTAL_DOCUMENT = 10_00_000
for idx, d in enumerate(dataset):
    write_to_file(idx, d["text"])

    if idx == TOTAL_DOCUMENT:
        break
