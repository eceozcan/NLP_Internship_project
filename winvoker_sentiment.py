# ================================
# Winvoker Veri Seti Ön İşleme + BERT Embedding (Colab GPU için)
# ================================

# Gerekli kütüphaneler

import pandas as pd
import re
from zemberek import TurkishMorphology
import joblib
import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset


# Zemberek Başlatma
morphology = TurkishMorphology.create_with_defaults()


# Veri Seti Yükleme
dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])


# Ön İşleme Fonksiyonu
def preprocess_text_zemberek(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    lemmas = []
    for w in words:
        try:
            result = morphology.lemmatize(w)
            if result:
                lemmas.append(result[0])
            else:
                lemmas.append(w)
        except:
            lemmas.append(w)
    return ' '.join(lemmas), lemmas

# Train ve Test setlerine uygulama
train_df['text'], train_df['tokens'] = zip(*train_df['text'].apply(preprocess_text_zemberek))
test_df['text'], test_df['tokens'] = zip(*test_df['text'].apply(preprocess_text_zemberek))

# İşlenmiş veriyi kaydetme
joblib.dump(train_df, 'processed_train_zemberek.joblib')
joblib.dump(test_df, 'processed_test_zemberek.joblib')

print("Zemberek ile ön işleme tamamlandı ve kaydedildi.")
print("Train set örnek:")
print(train_df.head(5))


# BERT Modeli Yükleme
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)
bert_model.to(device)


# BERT Embedding Fonksiyonu
def get_bert_embeddings(text_list, batch_size=16, max_len=128):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
        encoded = {key: val.to(device) for key, val in encoded.items()}

        with torch.no_grad():
            outputs = bert_model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # CLS token
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

print("Train için BERT embedding çıkarılıyor...")
train_embeddings = get_bert_embeddings(train_df['text'].tolist(), batch_size=16, max_len=128)

print("Test için BERT embedding çıkarılıyor...")
test_embeddings = get_bert_embeddings(test_df['text'].tolist(), batch_size=16, max_len=128)

# Kaydetme
joblib.dump(train_embeddings, "bert_train_embeddings.joblib")
joblib.dump(test_embeddings, "bert_test_embeddings.joblib")

print("BERT embeddingler kaydedildi.")
print("Train embedding shape:", train_embeddings.shape)
print("Test embedding shape:", test_embeddings.shape)
