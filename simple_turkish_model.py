import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# 1. Dataset (10k örnek)
texts_base = [
    # Pozitif örnekler
    "Bu ürün harika", "Çok memnun kaldım", "Tekrar alırım", 
    "Mükemmel deneyim", "Harika bir alışverişti",
    "Çok güzeldi 😂", "Harika hissediyorum 😍", "Süper ürün 👍", "Mutluyum 🤩", "Çok tatlı 🥰",

    # Negatif örnekler
    "Berbat bir deneyim", "Hiç beğenmedim", "Kötü kalite", 
    "Tavsiye etmiyorum", "Hayal kırıklığı oldu",
    "Çok kötü 😡", "Berbat hissediyorum 😭", "Hayal kırıklığı 🤬", "Beş para etmez 👎", "Asla tekrar almam 💔"
]

labels_base = [
    1, 1, 1, 1, 1,   # pozitif
    1, 1, 1, 1, 1,   # pozitif (emojili)
    0, 0, 0, 0, 0,   # negatif
    0, 0, 0, 0, 0    # negatif (emojili)
]


# 10k veri
texts = texts_base * 1000
labels = labels_base * 1000

df = pd.DataFrame({"text": texts, "label": labels})
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train_dataset.csv", index=False, encoding="utf-8-sig")
val_df.to_csv("val_dataset.csv", index=False, encoding="utf-8-sig")

# 2. Dataset class
class TextDataset(Dataset):
    def __init__(self, df, vocab=None):
        self.texts = df['text'].values
        self.labels = df['label'].values
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab
        self.encoded_texts = [self.encode(text) for text in self.texts]

    def build_vocab(self, texts):
        vocab = {"<PAD>":0, "<UNK>":1}
        idx = 2
        for sentence in texts:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx +=1
        return vocab

    def encode(self, text):
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_texts[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


# 3. Collate function for padding
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(x) for x in texts]
    max_len = max(lengths)
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, x in enumerate(texts):
        padded_texts[i,:len(x)] = x
    labels = torch.tensor(labels)
    return padded_texts, labels


# 4. Model
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


# 5. DataLoader
train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df, vocab=train_dataset.vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# 6. Model + BitFit
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleLSTM(len(train_dataset.vocab)).to(device)

# BitFit
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# 7. Training
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")


# 8. Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Validation Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")


# 9. Quantization-ready model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)

# Quantized model ile validation
all_preds_q, all_labels_q = [], []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = quantized_model(x)
        preds = torch.argmax(out, dim=1)
        all_preds_q.extend(preds.cpu().numpy())
        all_labels_q.extend(y.cpu().numpy())

acc_q = accuracy_score(all_labels_q, all_preds_q)
f1_q = f1_score(all_labels_q, all_preds_q, average="macro")
print(f"Quantized Model - Validation Accuracy: {acc_q:.4f}, Macro-F1: {f1_q:.4f}")


# 10. Save model
torch.save(model.state_dict(), "simple_lstm_bitfit.pth")
torch.save(quantized_model.state_dict(), "simple_lstm_bitfit_quantized.pth")
