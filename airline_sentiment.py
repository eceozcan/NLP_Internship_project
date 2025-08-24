# airlines tweet sentiment analysis

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModel
import joblib

# --- 0) NLTK veri paketlerini indirme ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- 1) Veriyi Yükleme ---
df = pd.read_csv("Tweets.csv")
print("Orijinal veri seti (ilk 10):")
print(df.head(10))

# --- 2) Ön İşleme ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Metni duygu analizi için temizler:
    - Küçük harfe çevirme
    - URL, mention, hashtag temizleme
    - Noktalama ve sayıları kaldırma
    - Tokenizasyon + stopword çıkarma
    - Lemmatization
    """
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# NaN değerleri boş stringe çevirme
df["text"] = df["text"].fillna("")
df["clean_text"] = df["text"].apply(clean_text)

print("\nTemizlenmiş veri seti (ilk 5):")
print(df[["text", "clean_text"]].head())

# Temizlenmiş CSV olarak kaydetme
df.to_csv("Tweets_clean.csv", index=False)

# --- 3) Vektörleştirme (BERT) ---
print("\nBERT modeli yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_bert_embeddings(text_list, batch_size=16, max_length=64):
  
    all_embeds = []
    model.eval()

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            # Mean pooling ile cümle embedding
            batch_emb = out.last_hidden_state.mean(dim=1).cpu()  # [B, 768]

        all_embeds.append(batch_emb)

    return torch.cat(all_embeds, dim=0)  # [N, 768]

# --- 4) Tüm veri seti embedding ---
print("\nTüm veri seti embedding çıkarılıyor...")
all_embeddings = get_bert_embeddings(df["clean_text"].tolist(), batch_size=16, max_length=64)
print("Tüm veri seti embedding boyutu:", all_embeddings.shape)

# --- 5) Embeddingleri kaydetme ---
joblib.dump(all_embeddings, "Tweets_BERT_embeddings.joblib")
print("BERT embeddingler 'Tweets_BERT_embeddings.joblib' olarak kaydedildi.")
