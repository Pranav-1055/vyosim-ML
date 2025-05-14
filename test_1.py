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

# Dataset definition
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

# Model definition
class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=3444):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable()  # Enable gradient checkpointing
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# Load data
path = '/content/merged.csv'
df = pd.read_csv(path)
label_cols = df.columns.difference(['text'])
df[label_cols] = df[label_cols].fillna(0)
df[label_cols] = df[label_cols].astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df[label_cols], test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Model, optimizer, loss, scheduler
model = BertMultiLabelClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 10

best_f1 = 0.0
start_epoch = 0
checkpoint_files = [f for f in os.listdir('.') if re.match(r'checkpoint_epoch_\d+\.pt', f)]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from checkpoint: {latest_checkpoint} (epoch {start_epoch})")

# Training loop with chunked dataloading to prevent RAM crash
chunk_size = 10000
num_chunks = int(np.ceil(len(train_texts) / chunk_size))
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(train_texts))
        chunk_texts = train_texts[start_idx:end_idx].tolist()
        chunk_labels = train_labels.values[start_idx:end_idx]

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

    print(f"Epoch {epoch+1}, Training Loss: {total_loss/(num_chunks):.4f}")

    # Save checkpoint
    checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Evaluation
    val_dataset = CustomDataset(val_texts.tolist(), val_labels.values, tokenizer, max_len=512)
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
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"New best model saved with F1: {best_f1:.4f}")
