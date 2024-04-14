import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
# from torch.optim import optimizer
from transformers import LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, \
    AutoModelForSequenceClassification, get_scheduler, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

# dataset = Dataset()
dataset = load_dataset("glue", "cola")
#dataset["train"][:1024]

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64)

small_dataset = dataset["train"].shuffle(seed=42).select(range(2048))
tokenized_datasets = small_dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])

train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=32)


# model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=2)
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# print(model)
# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id
model = model.half()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

#model = torch.nn.DataParallel(model, device_ids=[0])

model.to(device)

num_training_steps = len(train_dataloader)

progress_bar = tqdm(range(num_training_steps))
# for name, param in model.named_parameters():
#     print(name, param.dtype)
model.zero_grad()

optimizer = AdamW(model.parameters(), lr=1e-4)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
# model.train()
# for epoch in range(num_epochs):
loss_history = []

for batch in train_dataloader:
    # print(batch["input_ids"].shape, batch["labels"].shape)
    batch = {k: v.to(device) for k, v in batch.items()}
    # input = batch["input_ids"].to(device)
    # print(batch["input_ids"].shape)
    outputs = model(**batch)
    loss = outputs.loss
    loss_history.append(loss.item())
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

print(loss_history)

# del model
# gc.collect()
# torch.cuda.empty_cache()
