# Cell 1: Imports and Setup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW, get_scheduler
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import re
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Directory for saving checkpoints in Google Drive
checkpoint_dir = '/content/drive/MyDrive/bert_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Cell 2: Load and Preprocess Dataset in Chunks to Avoid RAM Crash
merged_path = '/content/drive/MyDrive/bert_checkpoints/merged.csv'

# Remove existing merged file if exists to avoid duplication
if os.path.exists(merged_path):
    os.remove(merged_path)

chunksize = 10000
for chunk in pd.read_csv('/content/test_model_data.csv', chunksize=chunksize):
    chunk['text'] = chunk[['html_title', 'h1', 'h2', 'p']].fillna('').agg(' '.join, axis=1)
    label_cols = chunk.columns.difference(['html_title', 'h1', 'h2', 'p', 'text'])
    chunk[label_cols] = chunk[label_cols].fillna(0).astype(int)
    chunk[['text'] + label_cols.tolist()].to_csv(merged_path, mode='a', index=False, header=not os.path.exists(merged_path))

# Cell 3: Dataset Definition
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Cell 4: Model Definition
class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=3444):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# Cell 5: Tokenizer, Data Split, Model & Optimizer Setup
df = pd.read_csv('/content/drive/MyDrive/bert_checkpoints/merged.csv')
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df[df.columns.difference(['text'])], test_size=0.2, random_state=42)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertMultiLabelClassifier(num_labels=train_labels.shape[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 10

# Load from checkpoint if available
best_f1 = 0.0
start_epoch = 0
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'checkpoint_epoch_\d+\.pt', f)]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'\\d+', x)[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from checkpoint: {latest_checkpoint} (epoch {start_epoch})")

# Cell 5.5: Load Chunk Checkpoint if Exists
chunk_resume_path = os.path.join(checkpoint_dir, 'last_chunk_checkpoint.pt')
resume_epoch, resume_chunk = start_epoch, 0
if os.path.exists(chunk_resume_path):
    chunk_checkpoint = torch.load(chunk_resume_path, map_location=device)
    model.load_state_dict(chunk_checkpoint['model_state_dict'])
    optimizer.load_state_dict(chunk_checkpoint['optimizer_state_dict'])
    resume_epoch = chunk_checkpoint['epoch']
    resume_chunk = chunk_checkpoint['chunk_id'] + 1
    print(f"Resuming training from Epoch {resume_epoch}, Chunk {resume_chunk}")

# Cell 6: Training Loop with Chunked Loading
chunk_size = 10000
num_chunks = int(np.ceil(len(train_texts) / chunk_size))
for epoch in range(resume_epoch, epochs):
    model.train()
    total_loss = 0

    for chunk_id in range(resume_chunk if epoch == resume_epoch else 0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(train_texts))
        chunk_texts = train_texts[start_idx:end_idx].tolist()
        chunk_labels = train_labels.values[start_idx:end_idx].astype(np.float32)


        train_dataset = CustomDataset(chunk_texts, chunk_labels, tokenizer, max_len=512)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

        total_steps = len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1} Chunk {chunk_id+1}/{num_chunks}")
            loop.set_postfix(loss=loss.item())

        del chunk_texts, chunk_labels, train_dataset, train_loader
        torch.cuda.empty_cache()

        # Save temporary chunk checkpoint
        torch.save({
            'epoch': epoch,
            'chunk_id': chunk_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, chunk_resume_path)
        print(f"Temporary checkpoint saved at epoch {epoch}, chunk {chunk_id}")

    print(f"Epoch {epoch+1}, Training Loss: {total_loss/(num_chunks):.4f}")

    # Save full epoch checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Evaluation
    val_dataset = CustomDataset(val_texts.tolist(), val_labels.values.astype(np.float32), tokenizer, max_len=512)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.3).int()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Validation F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
        print(f"New best model saved with F1: {best_f1:.4f}")
