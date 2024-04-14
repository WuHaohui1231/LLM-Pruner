from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=2)