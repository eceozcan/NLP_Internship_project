# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import TrainingArguments, Trainer
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
# from transformers import TrainingArguments

# # 1. Dataset yükle
# dataset = load_dataset(
#     "csv",
#     data_files={"train": "data/train.csv", "validation": "data/val.csv"}
# )

# # 2. Tokenizer ve model yükle
# model_name = "dbmdz/bert-base-turkish-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=2  # Pozitif / Negatif
# )

# # 3. Tokenization
# def tokenize_function(example):
#     return tokenizer(example["text"], padding="max_length", truncation=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # 4. Metric fonksiyonu
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     acc = accuracy_score(labels, predictions)
#     f1 = f1_score(labels, predictions, average="macro")
#     return {"accuracy": acc, "macro_f1": f1}

# # 5. Training args
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",   # ✔ yeni sürümde bu geçerli
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     save_total_limit=1,
# )


# # 6. Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# # 7. Train
# trainer.train()

import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# 1. Dataset
# -----------------------------
if not os.path.exists("data/train.csv") or not os.path.exists("data/val.csv"):
    raise FileNotFoundError("Lütfen önce create_synthetic.py ile veri oluşturun.")

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# -----------------------------
# 2. Tokenizer & Model
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(train_df['label'].unique())
)

# -----------------------------
# 3. LoRA Config
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 4. TrainingArguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results_lora",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    logging_dir="./logs_lora",
    logging_steps=50,
    save_total_limit=1,
    evaluation_strategy="steps",  # Bu parametre artık destekleniyor
    eval_steps=50,
    save_strategy="steps",
)

# -----------------------------
# 5. Metrics
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 7. Train
# -----------------------------
trainer.train()
