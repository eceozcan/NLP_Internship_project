import pandas as pd
import re
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Dataset yükleme
df = pd.read_csv("magaza_yorumlari_duygu_analizi.csv", encoding="utf-16")

# Kolon isimlerini temizleme
df.columns = df.columns.str.strip()

# 2. Emojileri metne çevirme (demojize)
def emoji_to_text(text):
    return emoji.demojize(str(text), language="tr")

df["Görüş_clean"] = df["Görüş"].apply(emoji_to_text)

# 3. Basit text temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # linkleri sil
    text = re.sub(r"[^a-zçğıöşü0-9\s:]", " ", text)  # noktalama hariç
    return text

df["Görüş_clean"] = df["Görüş_clean"].apply(clean_text)

# 4. İlk 100 yorumu göster
print("İlk 100 demojize edilmiş ve temizlenmiş yorum:")
print(df["Görüş_clean"].head(100))

# 5. Train-test ayırma
X_train, X_test, y_train, y_test = train_test_split(
    df["Görüş_clean"], df["Durum"], test_size=0.2, random_state=42
)

# 6. TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Model (Logistic Regression)
clf = LogisticRegression(max_iter=500)
clf.fit(X_train_vec, y_train)

# 8. Tahmin ve rapor
y_pred = clf.predict(X_test_vec)

# 9. Kaydetme
df.to_csv("ürün_yorumlar.csv", index=False, encoding="utf-16")


print("\nModel Performansı:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
