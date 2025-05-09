# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install necessary libraries
!pip install -q transformers datasets

# Step 3: Import packages
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# Step 4: Load your dataset
df = pd.read_csv("/content/drive/MyDrive/test_model_data.csv")

# Combine text columns
text_cols = ["html_title", "h1", "h2", "p"]
df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

# Identify label columns
label_cols = df.select_dtypes("int64").columns.tolist()

# Step 5: Convert to HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df[["text"] + label_cols])
hf_dataset = hf_dataset.train_test_split(test_size=0.1)

# Format labels
def format_labels(example):
    example["labels"] = [example[col] for col in label_cols]
    return example

hf_dataset = hf_dataset.map(format_labels)

# Step 6: Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

hf_dataset = hf_dataset.map(tokenize, batched=True)

# Step 7: Remove unnecessary columns
hf_dataset = hf_dataset.remove_columns(["text"] + label_cols)
hf_dataset.set_format("torch")

# Step 8: Define model
num_labels = len(label_cols)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# Step 9: Custom data collator for float labels
def custom_collate(batch):
    batch = {k: torch.stack([d[k] for d in batch]) for k in batch[0]}
    batch["labels"] = batch["labels"].float()
    return batch

# Step 10: Define TrainingArguments
output_dir = "/content/drive/MyDrive/bert_model_checkpoint"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    report_to="none"
)

# Step 11: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["test"],
    data_collator=custom_collate,
    tokenizer=tokenizer
)

trainer.train()
