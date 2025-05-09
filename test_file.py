from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers datasets

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# Step 1: Load the dataset in chunks
chunk_size = 10000  # You can adjust this based on available RAM
data_chunks = pd.read_csv("/content/drive/MyDrive/test_model_data.csv", chunksize=chunk_size)

# Combine text columns
text_cols = ["html_title", "h1", "h2", "p"]

# Step 2: Define function to process each chunk
def process_chunk(chunk):
    # Combine text columns into a single "text" column
    chunk["text"] = chunk[text_cols].fillna("").agg(" ".join, axis=1)
    
    # Identify label columns dynamically (assuming they are integers)
    label_cols = chunk.select_dtypes("int64").columns.tolist()
    
    # Convert to HuggingFace Dataset
    hf_chunk = Dataset.from_pandas(chunk[["text"] + label_cols])
    hf_chunk = hf_chunk.map(format_labels)  # Format labels
    hf_chunk = hf_chunk.map(tokenize, batched=True)  # Tokenize text
    hf_chunk.set_format("torch")  # Convert to torch tensors
    
    return hf_chunk

# Process the chunks and combine
hf_dataset_list = [process_chunk(chunk) for chunk in data_chunks]

# Combine all chunks into a single dataset
hf_dataset = DatasetDict({
    'train': hf_dataset_list[0],  # Assuming first chunk is for training
    'test': hf_dataset_list[1]    # Assuming second chunk is for testing (adjust as needed)
})

# Step 3: Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize function for batching
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# Step 4: Format labels function
def format_labels(example):
    label_cols = df.select_dtypes("int64").columns.tolist()
    example["labels"] = [example[col] for col in label_cols]
    return example

# Step 5: Define model
num_labels = len(label_cols)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# Step 6: Custom collate function
def custom_collate(batch):
    batch = {k: torch.stack([d[k] for d in batch]) for k in batch[0]}
    batch["labels"] = batch["labels"].float()
    return batch

# Step 7: Define TrainingArguments
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

# Step 8: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["test"],
    data_collator=custom_collate,
    tokenizer=tokenizer
)

trainer.train()
