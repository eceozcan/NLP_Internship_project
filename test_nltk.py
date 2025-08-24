# # ==============================================================
# # Winvoker Veri Seti Türkçe Duygu Analizi: Ön İşleme Scripti
# # ==============================================================

# # Gerekli kütüphaneleri import ediyoruz
# from datasets import load_dataset  # Hugging Face veri setini yüklemek için
# import pandas as pd                # DataFrame işlemleri için
# import re                          # Regex ile metin temizleme için
# import nltk                        # NLP araçları için
# nltk.download('stopwords')         # Stop-word listesini indiriyoruz
# from nltk.corpus import stopwords  # Stop-wordleri kullanmak için

# # ==============================================================
# # Stop-word listesini oluşturuyoruz (Türkçe)
# # ==============================================================

# stop_words = set(stopwords.words('turkish'))
# # Örnek: print(list(stop_words)[:20]) ile ilk 20 stop-word görülebilir

# # ==============================================================
# # Veri setini yükleme
# # ==============================================================

# dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")

# # Train ve Test setlerini pandas DataFrame'e çeviriyoruz
# train_df = pd.DataFrame(dataset['train'])
# test_df = pd.DataFrame(dataset['test'])

# print("Train set satır sayısı:", len(train_df))
# print("Test set satır sayısı:", len(test_df))

# # ==============================================================
# # Ön işleme Fonksiyonu
# # - Küçük harfe çevirme
# # - Noktalama işaretlerini temizleme
# # - Stop-word çıkarma
# # - Tokenizasyon
# # ==============================================================

# def preprocess_text(text):
#     # 1. Küçük harfe çevir
#     text = text.lower()
    
#     # 2. Noktalama ve özel karakterleri kaldır
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # 3. Stop-word çıkarma
#     text = ' '.join([w for w in text.split() if w not in stop_words])
    
#     # 4. Tokenizasyon (kelimelere ayırma)
#     tokens = text.split()
    
#     return text, tokens

# # ==============================================================
# # Train setine uygula
# # ==============================================================

# train_df['text'], train_df['tokens'] = zip(*train_df['text'].apply(preprocess_text))

# # ==============================================================
# # Test setine uygula
# # ==============================================================

# test_df['text'], test_df['tokens'] = zip(*test_df['text'].apply(preprocess_text))

# # ==============================================================
# # İşlenmiş veriyi kaydet (Joblib ile)
# # ==============================================================

# import joblib

# joblib.dump(train_df, 'processed_train_full.joblib')
# joblib.dump(test_df, 'processed_test_full.joblib')

# print("NLTK ile Ön işleme tamamlandı ve veri setleri kaydedildi.")
# print("Train set örnek:")
# print(train_df.head())
