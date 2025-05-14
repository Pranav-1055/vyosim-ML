from google.colab import drive
drive.mount('/content/drive')


import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import os, gc, numpy as np

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 8
NUM_LABELS = 3444
CHUNKSIZE = 1000
EPOCHS = 3
CSV_PATH = "/content/drive/MyDrive/test_model_data.csv"
START_CHUNK = 0  # set this to resume from specific chunk
RESUME_CHECKPOINT = None  # set to folder name if resuming, e.g., "checkpoint_epoch0_chunk7"
VAL_SPLIT = 0.1

if RESUME_CHECKPOINT:
    tokenizer = BertTokenizerFast.from_pretrained(RESUME_CHECKPOINT)
    model = BertForSequenceClassification.from_pretrained(RESUME_CHECKPOINT).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
else:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        problem_type="multi_label_classification",
        num_labels=NUM_LABELS
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = AdamW(model.parameters(), lr=2e-5)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def train_on_chunk(chunk_df):
    # Combine text columns
    texts = (chunk_df['html_title'].fillna('') + ' ' +
             chunk_df['h1'].fillna('') + ' ' +
             chunk_df['p'].fillna('') + ' ' +
             chunk_df['h2'].fillna('')).tolist()

    # Identify label columns (anything that's not a content column)
    content_cols = ["html_title", "p", "h1", "h2"]
    label_cols = [col for col in chunk_df.columns if col not in content_cols]
    print("Top 10 most frequent labels in this chunk:")
    print(chunk_df[label_cols].sum().sort_values(ascending=False).head(10))

    # Ensure labels are filled correctly (with 0 for missing values)
    labels = chunk_df[label_cols].fillna(0).astype(int).values.tolist()
    print(f"Labels shape: {np.array(labels).shape}")
    print("Distribution of number of active labels per row:")
    print(chunk_df[label_cols].sum(axis=1).describe())
    # Tokenize the text data
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)

    # Create the dataset and split it into training and validation sets
    dataset = TextDataset(encodings, labels)
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Prepare DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # Train the model
    model.train()
    for batch in tqdm(train_loader, desc="Training batch", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validate the model
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating batch", leave=False):
            labels = batch['labels'].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (logits > 0.1).astype(int)
            print(f"logits shape: {outputs.logits.shape}, labels shape: {batch['labels'].shape}")


            all_preds.extend(preds)
            all_labels.extend(labels)

    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    print(f"Validation F1 (micro): {f1:.4f}")

    del encodings, labels, dataset, train_loader, val_loader
    gc.collect()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    chunk_iter = pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(chunk_iter):
        if i < START_CHUNK:
            continue

        print(f"\n--- Processing chunk {i+1} ---")
        train_on_chunk(chunk)

        checkpoint_dir = f"checkpoint_epoch{epoch}_chunk{i}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        torch.cuda.empty_cache()
        gc.collect()

model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
